import torch
import chessgpt.model.config as config
from torch.utils.data import DataLoader
from chessgpt.model import transformer as t
from chessgpt.model.data import PGNDataset
from datetime import datetime

path = "data/Carlsen.pgn"

device = torch.device("mps")
dataset = PGNDataset(path, device, max_games=200)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
)

model = t.ChessModel()
model.to(device)
model.train()

loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(config.num_epochs):
    print(f"--- Epoch {epoch} ---")
    model.train()
    for i, (input, target) in enumerate(dataloader):
        optimizer.zero_grad()
        # input.to(device)
        # target.to(device)
        output = model(input)
        # Reshape output and target for CrossEntropyLoss
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target = input.view(-1)  # (batch*seq_len,)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if i % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(input)
                val_output = val_output.view(-1, val_output.size(-1))
                val_target = target.view(-1)
                val_loss = loss_fn(val_output, val_target)
                print(f"Epoch {epoch} | Round {i}: Loss = {loss.item():.5f} Validation Loss = {val_loss.item():.5f}")
            model.train()
    if epoch % 2 == 0 and config.max_games > 500:
        torch.save(model.state_dict(), f"model-{datetime.now(): %Y-%m-%d-%H-%M-%S}-epoch-{epoch}.pth")