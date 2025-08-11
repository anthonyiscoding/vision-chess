import chess.pgn

f = open("data/carlsen_randjelovic_1999.pgn", encoding="utf8")
game_pgn = chess.pgn.read_game(f)
board = game_pgn.board()
moves = [m.uci() for m in game_pgn.mainline_moves()]
print(moves)