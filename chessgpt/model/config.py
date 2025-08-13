# Small FF network for testing
emb_dim = 128
# out_dim = 128
hidden_dim = 256
num_heads = 12
num_kv_heads = 6
assert num_heads % num_heads == 0, "num_heads must be evenly divisible by num_kv_heads"
head_dim = emb_dim // num_heads
qkv_bias = True
max_seq_len = 4096
# TODO: Temporarily zero as naive tokenizer depends on all previous moves
attn_dropout = 0.0