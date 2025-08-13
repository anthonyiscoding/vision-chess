from chess import pgn
import torch
import chessgpt.model.config as config
from chessgpt.model import tokenizer, transformer as t

path = "data/Carlsen.pgn"
    
with open(path, "r", encoding="utf8") as f:
    game = pgn.read_game(f)

# Seperate array entry for each game/move set
device = torch.device("cpu")
model = t.ChessModel()
input = torch.tensor([tokenizer.encode(game)])

model.to(device)
input.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(config.num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    # Reshape output and target for CrossEntropyLoss
    output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
    target = input.view(-1)  # (batch*seq_len,)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    # print(f"Epoch {epoch}: Loss = {loss.item()}")

    # TODO: Validate against other data than input
    model.eval()
    with torch.no_grad():
        val_output = model(input)
        val_output = val_output.view(-1, val_output.size(-1))
        val_target = input.view(-1)
        val_loss = loss_fn(val_output, val_target)
        print(f"Epoch {epoch}: Loss = {loss.item()} Validation Loss = {val_loss.item()}")
