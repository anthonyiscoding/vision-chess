from vision.model.tokenizer import (
    to_embedding,
    from_embedding,
    generate_all_possible_moves,
)


def test_to_from():
    moves = generate_all_possible_moves()

    for m in moves:
        encoded = to_embedding(m)
        decoded = from_embedding(encoded)
        assert m == decoded
