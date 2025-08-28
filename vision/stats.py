import torch
import json
import types
from vision.model.transformer import ChessModel

model_path = "models/model-latest.pt"
model_config = "models/model-latest-config.json"
with open(model_config, "r") as f:
    config = json.load(f)

config = types.SimpleNamespace(**config["default"])
model = ChessModel(config)

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model_parameters = [p.numel() for p in model.parameters()]

dims_count = {}
for p in sorted(model_parameters, reverse=True):
    if p in dims_count:
        dims_count[p] += 1
    else:
        dims_count[p] = 1

total_params = sum(model_parameters)
print("Model config", model_config)
print("Model parameters", model_parameters)
print("Dimension counts", dims_count)
print(f"Total parameters: {total_params:,}")
