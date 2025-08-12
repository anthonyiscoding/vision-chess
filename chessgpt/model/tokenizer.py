from chess.pgn import Game

def to_embedding(move: str):
    assert len(move) == 4, "Move must be a string of length 4"

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    file_start = files.index(move[0]) + 1
    rank_start = ranks.index(move[1]) + 1
    file_end = files.index(move[2]) + 1
    rank_end = ranks.index(move[3]) + 1

    return file_start * rank_start * file_end * rank_end

def encode(game: Game):
    token_ids = []
    for m in game.mainline_moves():
        token_ids.append(to_embedding(m.uci()))

    return token_ids
