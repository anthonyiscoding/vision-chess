from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    emb_dim: int
    batch_size: int
    vocabulary_size: int
    num_epochs: int
    num_heads: int
    hidden_dim: int
    num_kv_heads: int
    head_dim: int
    qkv_bias: bool
    max_seq_len: int
    attn_dropout: float
    transformer_layers: int
    batch_limit: int | None
    save_model: bool
    learning_rate: float
