from chess.pgn import Game

# This can be reduced to a simple math function, just experimenting
def create_vocabulary():
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
    vocab = []
    val = 0

    for f in files:
        for r in ranks:
            for f_ in files:
                for r_ in ranks:
                    val += 1
                    move = f"{f}{r}{f_}{r_}"
                    embedding = to_embedding(move)
                    print(val, embedding)
                    # assert val == embedding 
                    vocab.append(embedding)
    
    print(vocab, len(vocab))

def to_embedding(move: str):
    assert len(move) == 4, "Move must be a string of length 4"

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    file_start = files.index(move[0]) + 1
    rank_start = ranks.index(move[1]) + 1
    file_end = files.index(move[2]) + 1
    rank_end = ranks.index(move[3]) + 1

    return file_start * rank_start * file_end * rank_end

create_vocabulary()

def tokenizer(game: Game):
    moves = [m.uci() for m in game.mainline_moves()]

    return moves
