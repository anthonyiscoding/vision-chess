import chess.pgn
from chessgpt.model.tokenizer import encode

f = open("data/carlsen_randjelovic_1999.pgn", encoding="utf8")
game_pgn = chess.pgn.read_game(f)
board = game_pgn.board()
moves = [m.uci() for m in game_pgn.mainline_moves()]
token_ids = encode(game_pgn)
assert len(moves) == len(token_ids)

for index, m in enumerate(moves):
    print(m, token_ids[index])