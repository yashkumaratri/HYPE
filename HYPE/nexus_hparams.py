from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class HYPEHyperParams(HyperParams):
    # Method
    layers: List[int]
    use_hodge = True  # Enable Hodge Laplacian processing
    use_simplicial_attention = True  # Enable topological attention
    simplicial_k = 0.5  # Persistence threshold analogy
    residual_scale = 0.3 # New edit strength (0-1)
    grad_threshold = 0.01 # Minimum gradient magnitude to allow updates
    use_hypercomplex = True  # Toggle hypercomplex updates
    hc_components = 4  # Quaternion-style (4 components)
    vector_dim = 512  # Must be divisible by 4 (e.g., 512/4=128)
    projection_dim = 512
    fact_token: str
    gnn_num_grad_steps: int
    gnn_loss_layer: int
    gnn_lr: float
    gnn_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    early_stopping_loss: float
    context_template_length_params: List[List[int]]
    subgraph_size: int
    get_repr_layer: int
    gnn_fact_token_strategy: str
    gnn_dim_factor: float
    gnn_attn_drop: float
    gnn_feat_drop: float
    compute_v_strategy: str
    use_predefined_context: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
