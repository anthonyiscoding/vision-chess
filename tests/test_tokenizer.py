from chess.pgn import Game, read_game
import chess
from io import StringIO

from vision.model.tokenizer import (
    to_embedding,
    from_embedding,
    generate_all_possible_moves,
    encode_game,
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
