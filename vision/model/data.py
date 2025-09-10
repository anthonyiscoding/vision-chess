import torch
import random
import numpy as np
from torch.utils.data import Dataset
from vision.model import tokenizer
from chess import pgn
from typing import Literal


GameStage = Literal["full", "early", "middle", "late"]


class NpyDataset(Dataset):

    def __init__(
        self,
        files: list[str],
        max_seq_len: int | None = None,
        min_seq_len: int = 10,
        stage: GameStage = "full",
        random_window=True,
    ):
        super().__init__()

        self.samples = []
        self.length = 0

        # TODO: Optimize this, very inefficient method currently
        for f in files:
            games: np.ndarray = np.load(f, mmap_mode="r")
            sample_count = len(games)
            game_start = 0

            # TODO: Double check that the math is mathing
            for i in range(0, sample_count):
                game = games[i]
                game_length = len(game)

                if game_length < min_seq_len:
                    continue

                if max_seq_len:
                    game_end = min(game_length, max_seq_len)
                else:
                    game_end = game_length

                # TODO: Try adjusting stages in dataloader by epoch, first early, then middle, then late, then full, then repeat
                # TODO: For UCI data it might be better to always have game_start = 0 because it's the same argument against dropout (no context)
                # TODO: I could also be entirely wrong and should experiment with game_start = n and dropout
                match stage:
                    case "early":
                        game_start = 0
                        game_end = 15
                    case "middle":
                        game_start = 16
                        game_end = 31
                    case "late":
                        game_start = 32
                        game_end = game_length

                if random_window:
                    # Randomly limit game length so we get games starting and ending at every position
                    game_start = random.randint(
                        game_start, max(game_start, game_end - min_seq_len)
                    )
                    game_end = random.randint(game_start + min_seq_len, game_end)

                if game_end > game_length:
                    game_end = game_length

                self.samples.append((f, i, game_start, game_end))
                self.length += 1

    def __len__(self):
        return self.length

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
