from chess.pgn import Game, read_game
import chess
from io import StringIO
import warnings

from vision.model.tokenizer import (
    to_embedding,
    from_embedding,
    generate_all_possible_moves,
    encode_game,
    encode_array,
    decode,
)


def test_to_from():
    moves = generate_all_possible_moves()

    for m in moves:
        encoded = to_embedding(m)
        decoded = from_embedding(encoded)
        assert m == decoded


def test_to_embedding_special_tokens():
    # Test special tokens
    assert to_embedding("<|sog|>") == [16]
    assert to_embedding("<|eog|>") == [17]
    assert to_embedding("<|unk|>") == [18]
    assert to_embedding("<|pad|>") == [19]


def test_to_embedding_single_characters():
    # Test files
    assert to_embedding("a") == [0]
    assert to_embedding("h") == [7]

    # Test ranks
    assert to_embedding("1") == [8]
    assert to_embedding("8") == [15]


def test_to_embedding_moves():
    # Test normal 4-character moves
    assert to_embedding("e2e4") == [4, 9, 4, 11]
    assert to_embedding("a1h8") == [0, 8, 7, 15]


def test_to_embedding_promotions():
    # Test that promotions are trimmed to 4 characters
    assert to_embedding("e7e8q") == [4, 14, 4, 15]
    assert to_embedding("a7a8n") == [0, 14, 0, 15]


def test_to_embedding_unknown_characters():
    # Test moves with unknown characters
    assert to_embedding("e2x4") == [18]  # Should return unknown token
    assert to_embedding("z1a2") == [18]  # Should return unknown token


def test_to_embedding_empty_string():
    # Test empty string
    assert to_embedding("") == []


def test_to_embedding_short_moves():
    # Test moves shorter than 4 characters
    assert to_embedding("e2") == [4, 9]
    assert to_embedding("a") == [0]


def test_from_embedding_single_int():
    # Test single integer input
    assert from_embedding(0) == "a"
    assert from_embedding(7) == "h"
    assert from_embedding(8) == "1"
    assert from_embedding(15) == "8"


def test_from_embedding_special_tokens():
    # Test special tokens
    assert from_embedding(16) == "<|sog|>"
    assert from_embedding(17) == "<|eog|>"
    assert from_embedding(18) == "<|unk|>"
    assert from_embedding(19) == "<|pad|>"


def test_from_embedding_list():
    # Test list of integers
    assert from_embedding([4, 9, 4, 11]) == "e2e4"
    assert from_embedding([0, 8, 7, 15]) == "a1h8"
    assert from_embedding([16]) == "<|sog|>"
    assert from_embedding([17]) == "<|eog|>"


def test_from_embedding_unknown_tokens():
    # Test unknown embedding values
    assert from_embedding(999) == "<|unk|>"
    assert from_embedding([999, 1000]) == "<|unk|><|unk|>"
    assert from_embedding([0, 999, 7]) == "a<|unk|>h"


def test_from_embedding_empty_list():
    # Test empty list
    assert from_embedding([]) == ""


def test_from_embedding_mixed_valid_invalid():
    # Test mix of valid and invalid embeddings
    assert from_embedding([0, 999, 8]) == "a<|unk|>1"
    assert from_embedding([16, 999, 17]) == "<|sog|><|unk|><|eog|>"


def test_encode_game():

    # Test empty game
    empty_game = Game()
    result = encode_game(empty_game)
    assert result == [16, 17]  # <|sog|> and <|eog|> only

    # Test game with a few moves
    pgn_string = StringIO("e2e4 b8c6 g1f3 d7d6 b1c3 g7g6 f1c4")
    game = read_game(pgn_string)

    assert game is not None, "Reading game failed"

    result = encode_game(game)
    print(result)
    expected = [16]  # <|sog|>
    expected.extend([4, 9, 4, 11])
    expected.extend([1, 15, 2, 13])
    expected.extend([6, 8, 5, 10])
    expected.extend([3, 14, 3, 13])
    expected.extend([1, 8, 2, 10])
    expected.extend([6, 14, 6, 13])
    expected.extend([5, 8, 2, 11])
    expected.extend([17])  # <|eog|>
    assert result == expected

    # Test game with promotion
    promotion_game = Game()
    board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")  # Pawn ready to promote
    promotion_game = Game.from_board(board)
    promotion_game.add_main_variation(chess.Move.from_uci("a7a8q"))

    result = encode_game(promotion_game)
    expected = [16]  # <|sog|>
    expected.extend([0, 14, 0, 15])  # a7a8 (promotion trimmed)
    expected.extend([17])  # <|eog|>
    assert result == expected


def test_encode_array_empty_list():
    # Test empty move list
    result = encode_array([])
    assert result == []


def test_encode_array_single_move():
    # Test single move
    result = encode_array(["e2e4"])
    expected = [4, 9, 4, 11]  # e2e4
    assert result == expected


def test_encode_array_multiple_moves():
    # Test multiple moves
    moves = ["e2e4", "e7e5", "g1f3"]
    result = encode_array(moves)
    expected = []
    expected.extend([4, 9, 4, 11])  # e2e4
    expected.extend([4, 14, 4, 12])  # e7e5
    expected.extend([6, 8, 5, 10])  # g1f3
    assert result == expected


def test_encode_array_special_tokens():
    # Test array with special tokens
    moves = ["<|sog|>", "e2e4", "<|eog|>"]
    result = encode_array(moves)
    expected = [16, 4, 9, 4, 11, 17]  # <|sog|>, e2e4, <|eog|>
    assert result == expected


def test_encode_array_with_promotions():
    # Test moves with promotions (should be trimmed)
    moves = ["a7a8q", "h2h1n"]
    result = encode_array(moves)
    expected = []
    expected.extend([0, 14, 0, 15])  # a7a8 (q trimmed)
    expected.extend([7, 9, 7, 8])  # h2h1 (n trimmed)
    assert result == expected


def test_encode_array_with_unknown_moves():
    # Test array with moves containing unknown characters
    moves = ["e2e4", "z9z9", "a1h8"]
    result = encode_array(moves)
    expected = []
    expected.extend([4, 9, 4, 11])  # e2e4
    expected.extend([18])  # z9z9 -> <|unk|>
    expected.extend([0, 8, 7, 15])  # a1h8
    assert result == expected


def test_encode_array_mixed_length_moves():
    # Test array with moves of different lengths
    moves = ["e2", "e2e4", "a"]
    result = encode_array(moves)
    expected = []
    expected.extend([4, 9])  # e2
    expected.extend([4, 9, 4, 11])  # e2e4
    expected.extend([0])  # a
    assert result == expected


def test_decode_empty_list():
    # Test empty token list
    result = decode([])
    assert result == []


def test_decode_single_special_token():
    # Test single special tokens
    assert decode([16]) == ["<|sog|>"]
    assert decode([17]) == ["<|eog|>"]
    assert decode([18]) == ["<|unk|>"]
    assert decode([19]) == ["<|pad|>"]


def test_decode_multiple_special_tokens():
    # Test multiple special tokens
    result = decode([16, 17])
    assert result == ["<|sog|>", "<|eog|>"]

    result = decode([16, 18, 17])
    assert result == ["<|sog|>", "<|unk|>", "<|eog|>"]


def test_decode_single_move():
    # Test single 4-character move
    result = decode([4, 9, 4, 11])  # e2e4
    assert result == ["e2e4"]

    result = decode([0, 8, 7, 15])  # a1h8
    assert result == ["a1h8"]


def test_decode_multiple_moves():
    # Test multiple 4-character moves
    token_ids = [4, 9, 4, 11, 4, 14, 4, 12]  # e2e4, e7e5
    result = decode(token_ids)
    assert result == ["e2e4", "e7e5"]


def test_decode_game_with_special_tokens():
    # Test complete game with start and end tokens
    token_ids = [16]  # <|sog|>
    token_ids.extend([4, 9, 4, 11])  # e2e4
    token_ids.extend([4, 14, 4, 12])  # e7e5
    token_ids.extend([17])  # <|eog|>

    result = decode(token_ids)
    assert result == ["<|sog|>", "e2e4", "e7e5", "<|eog|>"]


def test_decode_mixed_moves_and_special_tokens():
    # Test mix of moves and special tokens
    token_ids = [
        16,
        4,
        9,
        4,
        11,
        18,
        0,
        8,
        7,
        15,
        17,
    ]  # <|sog|>, e2e4, <|unk|>, a1h8, <|eog|>
    result = decode(token_ids)
    assert result == ["<|sog|>", "e2e4", "<|unk|>", "a1h8", "<|eog|>"]


def test_decode_incomplete_move_warning():
    # Test incomplete move at the end (should trigger warning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test with 1 token remaining
        result = decode([4, 9, 4, 11, 4])  # e2e4 + incomplete
        assert len(w) == 1
        assert "Incomplete move detected" in str(w[0].message)
        assert result == ["e2e4"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test with 2 tokens remaining
        result = decode([4, 9, 4, 11, 4, 9])  # e2e4 + incomplete
        assert len(w) == 1
        assert "Incomplete move detected" in str(w[0].message)
        assert result == ["e2e4"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test with 3 tokens remaining
        result = decode([4, 9, 4, 11, 4, 9, 4])  # e2e4 + incomplete
        assert len(w) == 1
        assert "Incomplete move detected" in str(w[0].message)
        assert result == ["e2e4"]


def test_decode_incomplete_move_after_special_token():
    # Test incomplete move after special token

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = decode([16, 4, 9, 4, 11, 4])  # <|sog|>, e2e4, incomplete
        assert len(w) == 1
        assert "Incomplete move detected" in str(w[0].message)
        assert result == ["<|sog|>", "e2e4"]


def test_decode_only_incomplete_move():
    # Test only incomplete move tokens

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = decode([4, 9])  # incomplete move
        assert len(w) == 1
        assert "Incomplete move detected" in str(w[0].message)
        assert result == []


def test_decode_unknown_embeddings_in_moves():
    # Test moves with unknown embeddings (should still decode but with <|unk|>)
    result = decode([999, 999, 999, 999])  # All unknown tokens forming a "move"
    assert result == ["<|unk|>", "<|unk|>", "<|unk|>", "<|unk|>"]


def test_decode_roundtrip():
    # Test encoding then decoding gives back original structure

    moves = ["<|sog|>", "e2e4", "e7e5", "g1f3", "<|eog|>"]
    encoded = encode_array(moves)
    decoded = decode(encoded)
    assert decoded == moves
