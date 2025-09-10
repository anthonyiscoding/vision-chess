import torch
import random
import numpy as np
from torch.utils.data import Dataset
from vision.model import tokenizer
from chess import pgn
from typing import Literal


GameStage = Literal["full", "early", "mid", "late"]


class NpyDataset(Dataset):

    def __init__(
        self,
        files: list[str],
        max_seq_len: int | None = None,
        stage: GameStage = "full",
        random_length=False,
    ):
        super().__init__()

        self.samples = []
        self.max_seq_len = max_seq_len

        # TODO: Optimize this and __len__, very inefficient method currently
        for f in files:
            games: np.ndarray = np.load(f, mmap_mode="r")
            sample_count = len(games)
            game_start = 0

            # TODO: Double check that the math is mathing
            for i in range(0, sample_count):
                game = games[i]
                if self.max_seq_len:
                    game_end = min(len(game), self.max_seq_len)
                else:
                    game_end = len(game)
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
                    if len(game) < 5:
                        continue
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
        #     return torch.tensor([], device="cpu"), torch.tensor([], device="cpu")

        input_ids = torch.tensor(token_ids[game_start : game_end - 1], device="cpu")
        target_ids = torch.tensor(token_ids[game_start + 1 : game_end], device="cpu")

        return input_ids, target_ids
