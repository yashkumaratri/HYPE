import torch
import dgl
from dgl.nn.pytorch.conv import EGATConv
from typing import List, Dict, Literal, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
from .repr_tools import get_hidden_state
from .HYPE_hparams import HYPEHyperParams
import torch.nn as nn
from geoopt import PoincareBall  # Add this import for hyperbolic geometry

# Initialize hyperbolic space with learnable curvature
hyperbolic_space = PoincareBall(c=1.0)  # c is the curvature parameter

def build_graph_from_triples(
        triples: list,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        hparams: HYPEHyperParams
) -> (dgl.DGLGraph(), list):

    # Create graph
    g = dgl.DGLGraph()
    g = g.to("cuda")

    nodes = {}
    edges = []
    node_features = []
    edge_features = []

    # Create relation type to index mapping
    relation_types = set(triple['relation'] for triple in triples)
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_types)}
    edge_type_ids = []  # Initialize edge type IDs

    # Create initial relation embeddings in hyperbolic space
    init_rel_emb = []

    for idx, relation in enumerate(relation_types):
        relation_vec = get_hidden_state(model, tokenizer, hparams, relation)
        # Project relation embeddings into hyperbolic space
        relation_vec_hyp = hyperbolic_space.expmap0(relation_vec.to("cuda"))
        init_rel_emb.append(relation_vec_hyp)

    # Stack relation embeddings
    init_rel_emb = torch.stack(init_rel_emb).to("cuda")

    print("Iterating Triples")

    for triple in triples[:hparams.subgraph_size]:
        subject_str = triple['subject']
        relation_str = triple['relation']
        target_str = triple['target']

        # Get subject and target embeddings
        subject_vec = get_hidden_state(model, tokenizer, hparams, subject_str)
        target_vec = get_hidden_state(model, tokenizer, hparams, target_str)

        # Project subject and target embeddings into hyperbolic space
        subject_vec_hyp = hyperbolic_space.expmap0(subject_vec.to("cuda"))
        target_vec_hyp = hyperbolic_space.expmap0(target_vec.to("cuda"))

        if subject_str not in nodes:
            nodes[subject_str] = subject_vec_hyp
        if target_str not in nodes:
            nodes[target_str] = target_vec_hyp

        edges.append((subject_str, target_str))

        # Use hyperbolic distance as edge feature
        relation_vec_hyp = init_rel_emb[relation_to_id[relation_str]]
        edge_features.append(relation_vec_hyp)

        edge_type_ids.append(relation_to_id[relation_str])

    # Sort nodes and create indices
    nodes_list = list(nodes.keys())
    nodes_list.sort()
    node_indices = {node: index for index, node in enumerate(nodes_list)}
    node_features = [nodes[node] for node in nodes_list]

    # Convert edges to indices
    edges = [(node_indices[v], node_indices[u]) for u, v in edges]

    # Add nodes and edges to the graph
    g.add_nodes(len(nodes_list))
    g.add_edges(*zip(*edges))

    # Add node and edge features
    g.ndata['feat'] = torch.stack(node_features).to("cuda")
    g.ndata['id'] = torch.tensor(
        [node_indices[n] for n in nodes_list], dtype=torch.long).to("cuda")
    g.edata['r_h'] = torch.stack(edge_features).to("cuda")
    g.edata['etype'] = torch.tensor(edge_type_ids, dtype=torch.long).to("cuda")

    # Add self-loops and normalize
    g = dgl.add_self_loop(g)
    indegrees = g.in_degrees().float()
    node_norm = torch.pow(indegrees, -1)
    g.ndata['norm'] = node_norm.view(-1, 1).to("cuda")

    print("Finished building graph")

    return g, node_indices, init_rel_emb