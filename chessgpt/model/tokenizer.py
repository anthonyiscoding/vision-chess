from chess.pgn import Game

# Naive implementation, encodes strings like "a1a1" which aren't valid moves.
def to_embedding(move: str):
    if move == '<|startofgame|>':
        return 4097

    if move == '<|endofgame|>':
        return 4098 

    if move == 'None' or move == None:
        return 0

    assert len(move) == 4, "Move must be a string of length 4"
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    file_start = files.index(move[0])
    rank_start = ranks.index(move[1])
    file_end = files.index(move[2])
    rank_end = ranks.index(move[3])

    return file_start + 8 * rank_start + 64 * file_end + 512 * rank_end + 1

def from_embedding(embedding: int) -> str:
    if embedding == 4097:
        return '<|startofgame|>'
    
    if embedding == 4098:
        return '<|endofgame|>'
    
    assert embedding != 0, "Embeddings should not contain 0. 0 should have been masked out in earlier steps."

    # Lookup is still zero indexed even though we won't encode embedding == 0. 
    embedding -= 1

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    file_start = embedding % 8
    embedding //= 8
    rank_start = embedding % 8
    embedding //= 8
    file_end = embedding % 8
    embedding //= 8
    rank_end = embedding % 8

    return files[file_start] + ranks[rank_start] + files[file_end] + ranks[rank_end]

def encode_game(game: Game):
    token_ids = []
    for m in game.mainline_moves():
        # [:4] is for trimming off promotions at the end of moves
        # TODO: Figure out if promotions are relevant to the model
        token_ids.append(to_embedding(m.uci()[:4]))

    return token_ids

def encode_array(move_list):
    token_ids = [to_embedding(m[:4]) for m in move_list]
    return token_ids

def decode(token_ids: list[int]):
    decoded_ids = []

    for t in token_ids:
        decoded_ids.append(t)
    
    return decoded_ids

def generate_all_possible_moves():
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