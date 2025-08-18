import argparse
import itertools
import random
from pathlib import Path
import chess.pgn
import numpy as np


def read_pgn(file: str):
    with open(file, mode="r") as f:
        game = chess.pgn.read_game(f)

        while game:
            moves = [m.uci() for m in game.mainline_moves()]
            yield moves
            game = chess.pgn.read_game(f)

    yield None


def write_np(
    move_collection, file_stem, output_dir, training_data_ratio=0.9, shuffle=True
):
    if shuffle:
        random.shuffle(move_collection)

    training_size = int(len(move_collection) * training_data_ratio)
    validation_size = len(move_collection) - training_size

    training_array = move_collection[:training_size]
    validation_array = move_collection[validation_size:]

    np.savez(f"{output_dir}/training/{file_stem}.npz", *training_array)
    np.savez(f"{output_dir}/validation/{file_stem}.npz", *validation_array)


def read_npz(file, n: str | bool):
    move_collection: np = np.load(file)

    if n is True:
        limit = len(move_collection)
    else:
        limit = min(len(move_collection), n)

    keys = move_collection.files[:limit]

    for k in keys:
        print(move_collection[k])


def pgn_to_npz(input, batch_size, output_dir):
    base_path = Path(input)
    file_glob = f"*.pgn"
    pgn_files = [f for f in base_path.glob(file_glob) if f.is_file()]

    for file in pgn_files:
        f = str(file)
        reader = read_pgn(f)
        move_list = [m for m in list(reader) if m is not None]

        loops = list(range(0, len(move_list), batch_size))
        for i in loops:
            start = i
            end = min(i + batch_size, len(move_list)) - 1
            try:
                write_np(
                    move_list[start : end],
                    f"{file.stem}-{start}-{end}",
                    output_dir,
                )
            except StopIteration:
                break


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

    if args.read and Path(args.input).is_dir():
        print("Input must be a single file when --read is provided")

    if args.read:
        read_npz(args.input, args.read)

    if not args.read:
        pgn_to_npz(args.input, args.batch_size, args.output_dir)
