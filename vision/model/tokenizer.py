from chess.pgn import Game
import warnings


# TODO: New encoding scheme. The UCI one may not encode enough meaning.
def _flip(dictionary: dict):

    flipped = {}
    for k, v in dictionary.items():
        if v not in dictionary:
            flipped[v] = k

    return flipped


files = ["a", "b", "c", "d", "e", "f", "g", "h"]
ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

base_vocab = files + ranks
token_to_embedding_map = {token: i for i, token in enumerate(base_vocab)}

# Add special tokens at the end
offset = len(token_to_embedding_map)
special_tokens_to_embeddings = {
    "<|sog|>": offset + 0,  # Start of game
    "<|eog|>": offset + 1,  # End of game
    "<|unk|>": offset + 2,  # Unknown token
    "<|pad|>": offset + 3,  # Padding
}
token_to_embedding_map.update(special_tokens_to_embeddings)

embedding_to_token_map = _flip(token_to_embedding_map)
VOCABULARY_SIZE = len(token_to_embedding_map)


def to_embedding(move: str):
    if move in token_to_embedding_map:
        return [token_to_embedding_map[move]]

    # [:4] is for trimming off promotions at the end of moves
    move = move[:4]
    try:
        return [token_to_embedding_map[char] for char in move]
    except KeyError:
        return [token_to_embedding_map["<|unk|>"]]


def from_embedding(embedding: list[int]) -> list[str]:
    if not isinstance(embedding, list):
        embedding = [embedding]
    return [embedding_to_token_map.get(i, "<|unk|>") for i in embedding]


def encode_game(game: Game) -> list[int]:
    token_ids: list[int] = []
    token_ids.extend(to_embedding("<|sog|>"))
    for m in game.mainline_moves():
        # TODO: Figure out if promotions are relevant to the model
        token_ids.extend(to_embedding(m.uci()))
    token_ids.extend(to_embedding("<|eog|>"))
    return token_ids


def encode_array(move_list):
    token_ids = []
    for m in move_list:
        token_ids.extend(to_embedding(m))
    return token_ids


def decode(token_ids: list[int]):
    moves = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] >= offset:  # Is a special token
            move = from_embedding(token_ids[i])
            moves.append("".join(move))
            i += 1
        else:  # It's a move
            if i + 4 <= len(token_ids):
                move = from_embedding(token_ids[i : i + 4])
                moves.append("".join(move))
                i += 4
            else:  # Incomplete move at the end
                warnings.warn(
                    f"Incomplete move detected, got {token_ids[i:]}, skipping rest of the list."
                )
                i = len(token_ids)
    return moves


def generate_all_possible_moves():
    # This function is less relevant with the new scheme, but we can update it
    return list(token_to_embedding_map.keys())
