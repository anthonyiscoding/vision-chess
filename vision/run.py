import argparse
import sys
import torch
from pathlib import Path
from vision.model import transformer as t
from vision.model import tokenizer
from vision.model.tokenizer import from_embedding
from vision.utils import get_device

parser = argparse.ArgumentParser(description="Run a pre-trained model")
parser.add_argument("model_path", type=str, help="Path to the pre-trained model")
args = parser.parse_args()

model_path = Path(args.model_path)

model = t.ChessModel.load_from_checkpoint(model_path)
device = get_device()
model.to(device)
model.stat

model.eval()


def main():
    print("Enter a move sequence (e.g. e2e4e7e5g1f3):", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        moves = [line[i : i + 4] for i in range(0, len(line), 4)]
        input_ids = tokenizer.encode_array(moves)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor)
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
