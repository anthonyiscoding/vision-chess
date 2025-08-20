import torch
import random
import numpy as np
from torch.utils.data import Dataset
from chessgpt.model import tokenizer, config
from chess import pgn
from typing import Literal


class PGNDataset(Dataset):
    def __init__(
        self, path, device, max_seq_len=config.max_seq_len, max_games=None, step=2
    ):
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

                token_ids = tokenizer.encode_game(game)

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

GameStage = Literal["full", "early", "mid", "late"]
class NpyDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        device,
        max_seq_len=config.max_seq_len,
        stage: GameStage = "full",
        random_length = False
        # step=1,
    ):
        super().__init__()

        self.samples = []
        self.device = device
        self.max_seq_len = max_seq_len
        # self.step = step

        # TODO: Optimize this and __len__, very inefficient method currently
        for f in files:
            games: np.ndarray = np.load(f, mmap_mode="r")
            sample_count = len(games)
            game_start = 0

            # TODO: Double check that the math is mathing
            for i in range(0, sample_count):
                game = games[i]
                game_end = min(len(game), self.max_seq_len)
                match stage:
                    case "early":
                        game_start = 0
                        game_end = 15
                    case "mid":
                        game_start = 16
                        game_end = 31 
                    case "late":
                        game_start = 32
                        game_end = 47

                if random_length:
                    if len(game) < 5: continue
                    # Randomly limit game length so we get games at every position
                    game_end = random.randint(game_start + 2, game_end)
                    
                self.samples.append((f, i, game_start, game_end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, i, game_start, game_end = self.samples[index]
        games = np.load(file_path, mmap_mode="r")
        game = games[i]

        token_ids = tokenizer.encode_array(game)

        # TODO: Skip games with unknown tokens
        # if token_ids.index(tokenizer.special_tokens_to_embeddings['<|unk|>']):
        #     return torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        input_ids = torch.tensor(token_ids[game_start:game_end - 1], device=self.device)
        target_ids = torch.tensor(token_ids[game_start + 1 : game_end], device=self.device)

        return input_ids, target_ids
