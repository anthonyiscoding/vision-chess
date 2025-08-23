from vision.model.tokenizer import generate_all_possible_moves

emb_dim = 64
batch_size = 1
vocabulary_size = len(generate_all_possible_moves())
num_epochs = 2
# out_dim = 128
hidden_dim = emb_dim * 2
num_heads = 1
num_kv_heads = num_heads
head_dim = emb_dim
qkv_bias = True
max_seq_len = 400
attn_dropout = 0.0
transformer_layers = 6
batch_limit = 250
save_model = False
learning_rate = 1e-3
