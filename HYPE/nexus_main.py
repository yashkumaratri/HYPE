import time
import torch
import dgl
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from .gnn import GNN

from .compute_u import compute_u
from .compute_v import compute_v
from .HYPE_hparams import HYPEHyperParams
from .build_graph import build_graph_from_triples
from .context import PREDEFINED_CONTEXT_TEMPLATES
# print("importing laplacins")
# from torch_geometric.utils import get_laplacian

from util import nethook
from util.generate import generate_fast
# from util.context import CONTEXT_TEMPLATES
CONTEXT_TEMPLATES_CACHE = None

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    Performs Möbius addition on vectors/tensors x and y in a space of curvature c.
    This implementation assumes x and y have the same shape.
    """
    print("DOInngggg mobius")
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    denominator = 1 + 2 * c * xy + c * y2 * x2
    result = ((1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y) / denominator
    return result

def apply_HYPE_to_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    requests: List[Dict[str, Union[List[str], str]]],
    hparams: HYPEHyperParams,
    graph_input: list,
    gnn_model: GNN,
    copy: Optional[bool] = False,
    return_orig_weights: Optional[bool] = False,
    cache_template: Optional[str] = None,
) -> Tuple[PreTrainedModel, Dict[str, torch.Tensor]]:

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    # 保存 RGCN 的初始权重
    initial_state_dict = gnn_model.state_dict()

    for i, (request, graph) in enumerate(zip(requests, graph_input)):

        deltas = execute_HYPE(model=model,
                              tokenizer=tokenizer,
                              request=request,
                              hparams=hparams,
                              gnn_model=gnn_model,
                              graph_triples=graph,
                              cache_template=(
                                  cache_template if i == 0 else None)
                              )

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                print("residual scale")
                upd_matrix = hparams.residual_scale * upd_matrix # residual scaling
                if hasattr(w, 'grad') and w.grad is not None:  # Safety check
                    grad_magnitudes = w.grad.abs().mean(dim=1)  # [out_features]
                    mask = (grad_magnitudes > hparams.grad_threshold).float().unsqueeze(1)
                    upd_matrix *= mask  # Apply gradient-based mask

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()
                
                w[...] = mobius_add(w, upd_matrix, c=0.1)

                # w[...] += upd_matrix

        # 恢复 RGCN 的参数
        gnn_model.load_state_dict(initial_state_dict)

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_HYPE(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict,
    hparams: HYPEHyperParams,
    gnn_model: GNN,
    graph_triples: list,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Executes the HYPE update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # g, node_indices, init_rel_emb = build_graph_from_triples(graph_triples, model, tokenizer, hparams)
    g, node_indices, init_rel_emb = build_graph_from_triples(
        graph_triples, model, tokenizer, hparams)
    # edge_index = torch.tensor([[s,t] for s,t,_ in graph_triples]).t().contiguous()
    # g.edge_index, g.edge_attr = get_laplacian(edge_index, normalization='sym')

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    if request["target_true"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_true"]["str"] = " " + request["target_true"]["str"]
    print(
        f"Executing HYPE algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    hparams.layers = sorted(hparams.layers)
    for layer in hparams.layers:
        left_vector, right_vector = None, None
        require_recompute = True

        # Retrieve k/v pair if already stored in cache
        # Must be first layer, since rewrites at previous layers affect future layers
        if layer == hparams.layers[0]:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    left_vector = torch.from_numpy(
                        data["left_vector"]).to("cuda")
                    right_vector = torch.from_numpy(
                        data["right_vector"]).to("cuda")
                    require_recompute = False
                except Exception as e:
                    print(
                        f"Error reading cache file due to {e}. Recomputing...")

        # Compute rank-1 update matrix
        left_vector: torch.Tensor = (
            left_vector
            if left_vector is not None
            else compute_u(
                model,
                tokenizer,
                request,
                hparams,
                layer,
                get_context_templates(
                    model, tokenizer, hparams.context_template_length_params, hparams
                ),
            )
        )
        print("Left vector shape:", left_vector.shape)

        right_vector: torch.Tensor = (
            right_vector
            if right_vector is not None
            else compute_v(
                model=model,
                tokenizer=tokenizer,
                request=request,
                hparams=hparams,
                layer=layer,
                left_vector=left_vector,  # Use fused embeddings here
                gnn_model=gnn_model,
                graph_input=g,
                node_indices=node_indices,
                init_rel_emb=init_rel_emb,
                context_templates=get_context_templates(
                    model, tokenizer, hparams.context_template_length_params, hparams
                ),
            )
        )
        print("Right vector shape:", right_vector.shape)
        print("loading simplicial")

        def simplicial_attention(x):
                return torch.sigmoid(torch.norm(x, dim=1) - 0.5)  # 0.5 is your persistence threshold
        attn = simplicial_attention(init_rel_emb)  # Use original node embeddings
        print("configuring right vector to simplicial")
        right_vector = (attn.unsqueeze(-1) * right_vector).sum(dim=0)

        # Cache vectors for future use
        if cache_fname is not None and require_recompute:
            cache_fname.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                cache_fname,
                **{
                    "left_vector": left_vector.detach().cpu().numpy(),
                    "right_vector": right_vector.detach().cpu().numpy(),
                },
            )
            print(f"Cached k/v pair at {cache_fname}")


        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(
                upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by HYPE does not match original weight shape. "
            "Check for bugs in the code?"
        )
    
def get_context_templates(model, tok, length_params, hparams):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        if hparams.use_predefined_context:
            CONTEXT_TEMPLATES_CACHE = PREDEFINED_CONTEXT_TEMPLATES

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE