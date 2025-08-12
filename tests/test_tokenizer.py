from chessgpt.model.tokenizer import to_embedding, from_embedding
def generate_all_moves():
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    moves = []

    for f in files:
        for r in ranks:
            for f_ in files:
                for r_ in ranks:
                    move = f"{f}{r}{f_}{r_}"
                    moves.append(move)
    
    moves.append('<|startofgame|>')
    return moves

def test_to_from():
    moves = generate_all_moves()

    for m in moves:
        encoded = to_embedding(m)
        decoded = from_embedding(encoded)
        assert m == decoded