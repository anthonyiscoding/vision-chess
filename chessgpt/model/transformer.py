import torch.nn as nn
from torchtune.modules import FeedForward, MultiHeadAttention, RotaryPositionalEmbeddings, KVCache
from torchtune.modules.transformer import TransformerSelfAttentionLayer
import chessgpt.model.config as config

input_dim = config.emb_dim
hidden_dim = config.hidden_dim
ff = FeedForward(
    gate_proj=nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim)),
    down_proj=nn.Linear(hidden_dim, input_dim),
)

# kv_cache = KVCache()

mha = MultiHeadAttention(
    embed_dim=config.emb_dim,
    num_heads=config.num_heads,
    num_kv_heads=config.num_kv_heads,
    q_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
    k_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
    v_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
    output_proj=nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias),
    pos_embeddings=RotaryPositionalEmbeddings(config.head_dim),
    max_seq_len=config.max_seq_len,
    attn_dropout=config.attn_dropout
)