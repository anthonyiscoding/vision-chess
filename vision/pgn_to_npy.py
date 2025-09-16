import argparse
from functools import reduce
import random
import sys
from pathlib import Path
import chess.pgn
from chess.pgn import Game
import numpy as np
from tqdm import tqdm
import os

from vision.model.tokenizer import encode_game, token_to_embedding_map


# TODO: Make this a proper class
class PgnProcessor:
    def __init__(self):
        self.total_tokens = 0
        self.total_games = 0

    def read_pgn_any(self, file: str):
        with open(file, mode="r") as f:
            game = True

            while game:
                try:
                    assert not isinstance(game, bool)
                    game = chess.pgn.read_game(f)
                    game_list = encode_game(game)
                    yield game_list
                except KeyboardInterrupt:
                    sys.exit(0)
                except:
                    continue

        return None  # TODO: Should probably raise StopIteration

    def write_np(
        self,
        move_collection,
        file_stem,
        output_dir,
        training_data_ratio=0.9,
        shuffle=True,
    ):
        training_dir = os.path.join(output_dir, "training")
        validation_dir = os.path.join(output_dir, "validation")
        os.makedirs(training_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)

        if shuffle:
            random.shuffle(move_collection)

        longest_array = reduce(
            lambda x, y: x if len(x) > len(y) else y, move_collection, []
        )
        padded_array = []

        for a in move_collection:
            padded_array.append(
                np.pad(
                    array=a,
                    pad_width=(0, len(longest_array) - len(a)),
                    mode="constant",
                    constant_values=token_to_embedding_map["<|pad|>"],
                )
            )

        split_at = int(len(padded_array) * training_data_ratio)

        training_array = padded_array[:split_at]
        validation_array = padded_array[split_at:]

        np.save(f"{output_dir}/training/{file_stem}.npy", training_array)
        np.save(f"{output_dir}/validation/{file_stem}.npy", validation_array)

    def read_npy(self, file, limit: int | None = None):
        move_collection: np.ndarray = np.load(file)

        if limit:
            l = min(len(move_collection), limit)
        else:
            l = len(move_collection)

        rows = move_collection[:l]

        for r in rows:
            print(r)

    def pgn_to_npy(self, input, batch_size, output_dir):
        pgn_files = self.list_pgn_files(input)

        with tqdm(pgn_files, desc="Processing PGN files", unit="file") as progress_bar:
            for file in progress_bar:
                f = str(file)
                reader = self.read_pgn_any(f)
                game_list = list(reader)
                game_list_length = len(game_list)

                progress_bar.set_description(
                    f"Processing {file.name} ({game_list_length} games)"
                )

                for i in range(0, game_list_length, batch_size):
                    start = i
                    end = min(i + batch_size, game_list_length) - 1
                    try:
                        self.write_np(
                            game_list[start:end],
                            f"{file.stem}-{start}-{end}",
                            output_dir,
                        )
                    except StopIteration:
                        break

                self.total_games += game_list_length
                progress_bar.set_postfix({"Total games:": self.total_games})

        return self.total_games

    def list_pgn_files(self, input):
        return self._list_files(input, f"*.pgn")

    def list_npy_files(self, input):
        return self._list_files(input, f"*.npy")

    def _list_files(self, input, file_glob):
        base_path = Path(input)
        if base_path.is_file():
            return [base_path]
        files = [f for f in base_path.glob(file_glob) if f.is_file()]
        return files


# TODO: Add progress bar for npy file writing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turn PGN files into NumPy arrays for processing"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data",
        help="A file or path containing .pgn files",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Maximum number of games per file", default=2000
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data", help="Directory to save np files"
    )
    parser.add_argument(
        "--read",
        nargs="?",
        const=True,
        default=False,
        type=int,
        help="Print the contents of npz file to stdout. If an integer is provided it only prints the first n records.",
    )
    args = parser.parse_args()
    input = Path(args.input)
    output_dir = Path(args.output_dir)
    pgn_processor = PgnProcessor()

    if args.read and input.is_dir():
        print("Input must be a single file when --read is provided")

    if args.read:
        pgn_processor.read_npy(input, args.read)

    if not args.read:
        total_games = pgn_processor.pgn_to_npy(input, args.batch_size, output_dir)
        print(f"Processed {total_games} total games")
