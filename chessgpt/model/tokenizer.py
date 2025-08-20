from chess.pgn import Game


def _flip(dictionary: dict):

   flipped = {}
   for k, v in dictionary.items():
       if v not in dictionary:
           flipped[v] = k  
    
   return flipped

special_tokens_to_embeddings = {
    "<|startofgame|>": 4096,
    "<|endofgame|>": 4097,
    "<|unk|>": 4098,
    "<|pad|>": 4099,
}

special_embeddings_to_tokens = _flip(special_tokens_to_embeddings)

# Naive implementation, encodes strings like "a1a1" which aren't valid moves.
# TODO: Improve move detection
def to_embedding(move: str):

    if move in special_tokens_to_embeddings.keys():
        return special_tokens_to_embeddings[move]

    assert len(move) == 4, "Move must be a string of length 4"

    try:
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

        file_start = files.index(move[0])
        rank_start = ranks.index(move[1])
        file_end = files.index(move[2])
        rank_end = ranks.index(move[3])

        return file_start + 8 * rank_start + 64 * file_end + 512 * rank_end
    except:
        return special_tokens_to_embeddings['<|unk|>']


def from_embedding(embedding: int) -> str:

    if embedding in special_embeddings_to_tokens.keys():
        return special_embeddings_to_tokens[embedding]

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

    for k in special_tokens_to_embeddings.keys():
        moves.append(k)
    
    return moves
