import torch
import torch.nn as nn
from torchtune.modules import (
    FeedForward,
    MultiHeadAttention,
    RotaryPositionalEmbeddings,
)
from torchtune.modules.transformer import TransformerSelfAttentionLayer


class ChessModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.emb_dim
        hidden_dim = config.hidden_dim
        ff = FeedForward(
            gate_proj=nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
            ),
            down_proj=nn.Linear(hidden_dim, input_dim),
        )

        # kv_cache = KVCache()

        mha = MultiHeadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            q_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
            k_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
            v_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
            output_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
            pos_embeddings=RotaryPositionalEmbeddings(config.head_dim),
            max_seq_len=config.max_seq_len,
            attn_dropout=config.attn_dropout,
        )

        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.emb_dim, padding_idx=0
        )
        self.positional_embedding = nn.Embedding(config.max_seq_len, config.emb_dim)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerSelfAttentionLayer(attn=mha, mlp=ff)
                for _ in range(config.transformer_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocabulary_size, bias=False)

    def forward(self, idx):
        _, seq_len = idx.shape
        token_embeds = self.token_embedding(idx)
        positional_embeds = self.positional_embedding(
            torch.arange(seq_len, device=idx.device)
        )

        x = token_embeds + positional_embeds
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)  # logits

        return x
