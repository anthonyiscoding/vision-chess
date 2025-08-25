from dataclasses import dataclass, field
from vision.model.config import Config
from vision.model.tokenizer import generate_all_possible_moves


@dataclass(frozen=True)
class TestConfig(Config):
    emb_dim: int = 64
    batch_size: int = 1
    vocabulary_size: int = field(
        default_factory=lambda: len(generate_all_possible_moves())
    )
    num_epochs: int = 2
    num_heads: int = 8
    hidden_dim: int = field(init=False)
    num_kv_heads: int = field(init=False)
    head_dim: int = field(init=False)
    qkv_bias: bool = True
    max_seq_len: int = 400  # TODO: Figure out reasonable max sequence length
    attn_dropout: float = 0.0
    transformer_layers: int = 6
    batch_limit: int | None = None
    save_model: bool = False
    learning_rate: float = 0.000374

    def __post_init__(self):
        object.__setattr__(self, "hidden_dim", self.emb_dim * 2)
        object.__setattr__(self, "num_kv_heads", self.num_heads)
        object.__setattr__(self, "head_dim", self.emb_dim // self.num_heads)


config = TestConfig()
