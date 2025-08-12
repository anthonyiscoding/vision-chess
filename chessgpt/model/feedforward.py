import torch.nn as nn
from torchtune.modules import FeedForward
import chessgpt.model.config as config

input_dim = config.emb_dim
hidden_dim = config.hidden_dim
ff = FeedForward(
    gate_proj=nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim)),
    down_proj=nn.Linear(hidden_dim, input_dim),
)