import argparse
import sys
import torch
from pathlib import Path
from vision.model import transformer as t
from vision.model import tokenizer
from vision.model.tokenizer import from_embedding, to_embedding
from vision.utils import get_device

parser = argparse.ArgumentParser(description="Run a pre-trained model")
parser.add_argument("model_path", type=str, help="Path to the pre-trained model")
args = parser.parse_args()

model_path = Path(args.model_path)

model = t.ChessModel.load_from_checkpoint(model_path)
device = get_device()
model.to(device)
model.eval()


def generate_next_move(input_ids: list[int], skip_moves: set = None) -> str:
    """Generate the next complete move (4 tokens) given the input sequence.

    Args:
        input_ids: The current game token sequence
        skip_moves: Set of move strings to skip (for getting alternatives)
    """
    if skip_moves is None:
        skip_moves = set()

    current_ids = input_ids.copy()
    generated_tokens = []

    # Generate 4 tokens to form a complete move
    for token_pos in range(4):
        input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor)
            next_logits = logits[0, -1]  # Get logits for the last position

            # Get top tokens sorted by probability
            top_tokens = torch.argsort(next_logits, descending=True)

            # For first token, check if we need to skip certain moves
            if token_pos == 0 and skip_moves:
                for token_idx in top_tokens:
                    token = token_idx.item()
                    # Try this token and generate the full move to check if we should skip it
                    temp_tokens = [token]
                    temp_ids = current_ids + temp_tokens

                    # Generate remaining 3 tokens for this candidate
                    for remaining_pos in range(1, 4):
                        temp_tensor = torch.tensor(
                            [temp_ids], dtype=torch.long, device=device
                        )
                        with torch.no_grad():
                            temp_logits = model(temp_tensor)
                            temp_next_logits = temp_logits[0, -1]
                            temp_next_token = torch.argmax(temp_next_logits).item()
                            temp_tokens.append(temp_next_token)
                            temp_ids.append(temp_next_token)

                    # Check if this complete move should be skipped
                    temp_move_chars = from_embedding(temp_tokens)
                    temp_move = "".join(temp_move_chars)

                    if temp_move not in skip_moves:
                        next_token = token
                        break
                else:
                    # If all moves are in skip_moves, just take the first one
                    next_token = top_tokens[0].item()
            else:
                # Normal case: take the most likely token
                next_token = torch.argmax(next_logits).item()

            generated_tokens.append(next_token)
            current_ids.append(next_token)

    # Convert the 4 generated tokens back to a move string
    move_chars = from_embedding(generated_tokens)
    return "".join(move_chars)


def print_game_state(moves: list[str]):
    """Print the current game state in a readable format."""
    if not moves:
        print("Game started.", file=sys.stderr)
        return

    print("Current game:", file=sys.stderr)
    for i, move in enumerate(moves):
        if i % 2 == 0:
            move_num = (i // 2) + 1
            print(f"{move_num}. {move}", end="", file=sys.stderr)
        else:
            print(f" {move}", file=sys.stderr)

    if len(moves) % 2 == 1:
        print("", file=sys.stderr)  # New line if last move was white's


def main():
    moves_played = []  # Keep track of all moves in the game
    ai_generated_moves = set()  # Track moves AI has already suggested

    # Start with the start-of-game token
    game_tokens = to_embedding("<|sog|>")

    print("Chess AI - Enter moves in UCI format (e.g., 'e2e4')", file=sys.stderr)
    print("Commands:", file=sys.stderr)
    print("  'quit' - Exit the game", file=sys.stderr)
    print("  'next' - Generate AI's next most likely move", file=sys.stderr)
    print("  'alternatives' - Show top 5 move alternatives", file=sys.stderr)
    print_game_state(moves_played)
    print("Enter your move:", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        if line.lower() == "quit":
            break

        if line.lower() == "next":
            try:
                # Generate next move, skipping ones already suggested
                ai_move = generate_next_move(game_tokens, ai_generated_moves)
                ai_generated_moves.add(ai_move)

                print(f"AI suggests: {ai_move}", file=sys.stderr)
                print(
                    "Enter this move or type 'next' for another suggestion:",
                    file=sys.stderr,
                )
                continue

            except Exception as e:
                print(f"Error generating move: {e}", file=sys.stderr)
                print("Enter your move:", file=sys.stderr)
                continue

        if line.lower() == "alternatives":
            try:
                # Show top 5 alternatives for the current position
                input_tensor = torch.tensor(
                    [game_tokens], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    logits = model(input_tensor)
                    next_logits = logits[0, -1]
                    top5 = torch.topk(next_logits, 5)
                    top5_tokens = top5.indices.tolist()
                    top5_scores = top5.values.tolist()

                    print("Top 5 first token alternatives:", file=sys.stderr)
                    for i, (token, score) in enumerate(zip(top5_tokens, top5_scores)):
                        token_char = from_embedding([token])[0]
                        print(f"  {i+1}. '{token_char}': {score:.3f}", file=sys.stderr)

                print("Enter your move:", file=sys.stderr)
                continue

            except Exception as e:
                print(f"Error showing alternatives: {e}", file=sys.stderr)
                print("Enter your move:", file=sys.stderr)
                continue

        # Validate move format (basic check)
        if len(line) != 4:
            print("Invalid move format. Use UCI format like 'e2e4'", file=sys.stderr)
            print("Or use 'next', 'alternatives', or 'quit'", file=sys.stderr)
            continue

        try:
            # Encode the user's move and add to game
            user_move_tokens = to_embedding(line)
            game_tokens.extend(user_move_tokens)
            moves_played.append(line)

            print(f"You played: {line}", file=sys.stderr)

            # Clear the AI generated moves set since position changed
            ai_generated_moves.clear()

            # Generate AI's response
            ai_move = generate_next_move(game_tokens)
            ai_generated_moves.add(ai_move)

            # Add AI's move to game
            ai_move_tokens = to_embedding(ai_move)
            game_tokens.extend(ai_move_tokens)
            moves_played.append(ai_move)

            print(f"AI plays: {ai_move}", file=sys.stderr)
            print_game_state(moves_played)

            print(
                "\nEnter your next move (or 'next' for AI suggestion):", file=sys.stderr
            )

        except Exception as e:
            print(f"Error processing move: {e}", file=sys.stderr)
            print("Enter your move:", file=sys.stderr)


if __name__ == "__main__":
    main()
