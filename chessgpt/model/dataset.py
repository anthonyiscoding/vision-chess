import torch
from torch.utils.data import Dataset, DataLoader
from chessgpt.model.tokenizer import tokenizer
from chess import pgn

class PGNDataset(Dataset):
    def __init__(self, path, max_length, stride):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        with open(path, mode="r", encoding="utf8") as f:
            game = pgn.read_game(f)
            token_ids = tokenizer(game)