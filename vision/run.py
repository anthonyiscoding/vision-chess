import argparse
import json
import sys
import torch
from pathlib import Path
from vision.model import transformer as t
from vision.model import tokenizer
from vision.model.config.default import Config
from vision.model.tokenizer import from_embedding

parser = argparse.ArgumentParser(description="Run a pre-trained model")
parser.add_argument("model_path", type=str, help="Path to the pre-trained model")
args = parser.parse_args()

model_path = Path(args.model_path)

model_config = Path(f"{model_path.parent}/{model_path.stem}-config.json")
with open(model_config, "r") as f:
    config_dict = json.load(f)
    config = Config(**config_dict)

device = torch.device("mps")
model = t.ChessModel(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def main():
    print("Enter a move sequence (e.g. e2e4e7e5g1f3):", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Split input into 4-char moves
        moves = [line[i : i + 4] for i in range(0, len(line), 4)]
        input_ids = tokenizer.encode_array(moves)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)  # batch size 1
        with torch.no_grad():
            logits = model(input_tensor)
            # Get logits for last position
            next_logits = logits[0, -1]
            print(f"Logits: {next_logits}")
            top5 = torch.topk(next_logits, 5)
            top5_tokens = top5.indices.tolist()
            top5_scores = top5.values.tolist()
            top5_moves = [from_embedding(token) for token in top5_tokens]
            print("Top 5 predicted next moves:")
            for move, score in zip(top5_moves, top5_scores):
                print(f"  {move}: {score}")
        print("Enter a move sequence (e.g. e2e4e7e5g1f3):", file=sys.stderr)


if __name__ == "__main__":
    main()
