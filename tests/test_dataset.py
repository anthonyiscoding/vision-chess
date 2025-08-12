from chessgpt.model.dataset import PGNDataset
from pprint import pp
from chess import pgn

def test_dataset():
    path = "data/carlsen_randjelovic_1999.pgn"
    dataset = PGNDataset(path)
    
    with open(path, "r", encoding="utf8") as f:
        game = pgn.read_game(f)
        moves = [m.uci() for m in game.mainline_moves()]

    assert len(dataset.input_ids[0]) == len(moves)