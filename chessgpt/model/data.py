import torch
from torch.utils.data import Dataset
from chessgpt.model import tokenizer
from chess import pgn

# TODO: Determine what a good default sequence length is
class PGNDataset(Dataset):
    def __init__(self, path, max_seq_len=20):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        with open(path, mode="r", encoding="utf8") as f:
            game = pgn.read_game(f)

            while game is not None:
                # print(f"Reading game: {game.headers["Event"]}, {game.headers["Date"]}, {game.headers["Round"]}")

                token_ids = tokenizer.encode(game)
            
                for i in range(0, len(token_ids) - max_seq_len):
                    input_chunk = token_ids[i:i + max_seq_len]
                    target_chunk = token_ids[i + 1: i + max_seq_len + 1]

                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))

                game = pgn.read_game(f)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]