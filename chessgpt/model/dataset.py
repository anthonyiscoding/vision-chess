import torch
from torch.utils.data import Dataset, DataLoader
from chessgpt.model import tokenizer
from chess import pgn

class PGNDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        with open(path, mode="r", encoding="utf8") as f:
            game = pgn.read_game(f)
            token_ids = tokenizer.encode(game)
        
        for i in range(0, len(token_ids)):
            input_chunk = token_ids[i:i] 
            target_chunk = token_ids[i + 1: i + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))