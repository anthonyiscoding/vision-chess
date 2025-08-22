from vision.model.tokenizer import generate_all_possible_moves

# TODO: Better config file management

# Small FF network for testing
emb_dim = 2048
batch_size = 32
# batch_size = 4 with emb_dim = 1024 seems to have the quickest reduction in running loss (8 -> 2 in 100 batches)
# versus batch_size = 8 (8 -> 4 in 50 batches), (8 -> 4 in 100 too) and stabilizing after
# End of epoch 0 it reaches loss of ~2.6 and stops learning (loss steady)
# Validation loss curve matches learning loss

# batch_size = 4 with emb_dim = 2048 is (8 -> 10 -> 3.7 in 100 batches)
# TODO: I just need to use an optimization library to run tests

# batch_size = emb_dim / 64 # RAM constraints on my machine, ~36GB RAM (w/ only 1 worker)
vocabulary_size = len(generate_all_possible_moves())
num_epochs = 8
# out_dim = 128
hidden_dim = emb_dim * 2
num_heads = 1
num_kv_heads = num_heads
# assert num_heads % num_kv_heads == 0, "num_heads must be evenly divisible by num_kv_heads"
head_dim = emb_dim
qkv_bias = True
max_seq_len = 400  # TODO: Figure out reasonable max sequence length
# TODO: Temporarily zero as naive tokenizer depends on all previous moves
attn_dropout = 0.0
transformer_layers = 6
batch_limit = None
save_model = True
