import torch
import torch.nn as nn
from torchtune.modules import FeedForward


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.scale * normed


class PreNormTransformerLayer(nn.Module):
    def __init__(self, attn, mlp, dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = attn
        self.norm2 = RMSNorm(dim)
        self.mlp = mlp

    def forward(self, x):
        # residual scaling to avoid variance blowup
        normed_x = self.norm1(x)
        # PyTorch MultiheadAttention expects (query, key, value) and returns (output, weights)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + (attn_output / (2**0.5))
        x = x + (self.mlp(self.norm2(x)) / (2**0.5))
        return x


class ChessModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.emb_dim
        hidden_dim = config.hidden_dim

        ff_config = {
            "gate_proj": nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU()),
            "down_proj": nn.Linear(hidden_dim, input_dim),
        }

        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.emb_dim, padding_idx=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                PreNormTransformerLayer(
                    attn=nn.MultiheadAttention(
                        embed_dim=config.emb_dim,
                        num_heads=config.num_heads,
                        dropout=config.attn_dropout,
                        bias=config.qkv_bias,
                        batch_first=True,
                    ),
                    mlp=FeedForward(**ff_config),
                    dim=config.emb_dim,
                )
                for _ in range(config.transformer_layers)
            ]
        )

        self.final_norm = RMSNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocabulary_size, bias=False)

    def forward(self, idx):
        x = self.token_embedding(idx)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.out_head(x)
