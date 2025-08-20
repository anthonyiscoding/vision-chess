from chessgpt.model.tokenizer import generate_all_possible_moves

# Small FF network for testing
emb_dim = 1024
batch_size = 4
# batch_size = emb_dim / 64 # RAM constraints on my machine, ~36GB RAM
vocabulary_size = len(generate_all_possible_moves())
num_epochs = 8
# out_dim = 128
hidden_dim = emb_dim * 2 
num_heads = 1
num_kv_heads = num_heads
# assert num_heads % num_kv_heads == 0, "num_heads must be evenly divisible by num_kv_heads"
head_dim = emb_dim 
qkv_bias = True
max_seq_len = 400 # TODO: Figure out reasonable max sequence length
# TODO: Temporarily zero as naive tokenizer depends on all previous moves
attn_dropout = 0.0
transformer_layers = 6
# max_games = 2500