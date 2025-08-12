from chess.pgn import Game
def tokenizer(game: Game):
    tokens = [m.uci() for m in game.mainline_moves()]

    return tokens