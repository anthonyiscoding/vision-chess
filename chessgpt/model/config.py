from chessgpt.model.tokenizer import generate_all_possible_moves

# Small FF network for testing
emb_dim = 128
# out_dim = 128
hidden_dim = 256
num_heads = 1
num_kv_heads = num_heads
# assert num_heads % num_kv_heads == 0, "num_heads must be evenly divisible by num_kv_heads"
head_dim = emb_dim 
qkv_bias = True
max_seq_len = 4096
# TODO: Temporarily zero as naive tokenizer depends on all previous moves
attn_dropout = 0.0
transformer_layers = 6
vocabulary_size = len(generate_all_possible_moves())
num_epochs = 3