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

def write_np(move_collection, file_stem, output_dir, training_data_ratio=0.9, shuffle=True):
    if shuffle:
        random.shuffle(move_collection)

    training_size = int(len(move_collection) * training_data_ratio)
    validation_size = len(move_collection) - training_size

    training_array = move_collection[:training_size]
    validation_array = move_collection[validation_size:]

    np.savez(f"{output_dir}/training/{file_stem}.npz", *training_array)
    np.savez(f"{output_dir}/validation/{file_stem}.npz", *validation_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn PGN files into NumPy arrays for processing") 
    parser.add_argument("--input", type=str, default ="./data", help="A path containing .pgn files")
    parser.add_argument("--batch-size", type=int, help="Maximum number of games per file", default=2000)
    parser.add_argument("--output-dir", type=str, default="./data", help="Directory to save np files")
    args = parser.parse_args()

    base_path = Path(args.input)
    file_glob = f"*.pgn"
    pgn_files = [f for f in base_path.glob(file_glob) if f.is_file()]
    
    for i, file in enumerate(pgn_files):
        f = str(file)
        reader = read_pgn(f)
        move_list = list(itertools.islice(reader, args.batch_size)) 

        while len(move_list):
            try:
                move_list = np.array(move_list, dtype=object)
                write_np(move_list, f"{file.stem}-{i}", args.output_dir)
                move_list = list(itertools.islice(reader, args.batch_size))
            except StopIteration:
                break
