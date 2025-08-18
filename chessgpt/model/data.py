import torch
from torch.utils.data import Dataset
from chessgpt.model import tokenizer, config
from chess import pgn


# TODO: Determine what a good default sequence length is
class PGNDataset(Dataset):
    def __init__(self, path, device, max_seq_len=config.max_seq_len, max_games=None, step=2):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        game_count = 0
        with open(path, mode="r", encoding="utf8") as f:
            game = pgn.read_game(f)

            while game:
                if max_games:
                    game_count += 1
                if max_games and game_count > max_games:
                    break
                # print(f"Reading game: {game.headers["Event"]}, {game.headers["Date"]}, {game.headers["Round"]}")

                token_ids = tokenizer.encode(game)

                for i in range(0, len(token_ids) - max_seq_len, step):
                    input_chunk = token_ids[i : i + max_seq_len]
                    target_chunk = token_ids[i + 1 : i + max_seq_len + 1]

                    self.input_ids.append(torch.tensor(input_chunk, device=device))
                    self.target_ids.append(torch.tensor(target_chunk, device=device))

                game = pgn.read_game(f)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
