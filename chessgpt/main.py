import torch
import chessgpt.model.config as config
from torch.utils.data import DataLoader, random_split
from chessgpt.model import transformer as t
from chessgpt.model.data import PGNDataset, NPZDataset
from chessgpt.pgn_to_npy import list_npy_files
from datetime import datetime

path = "data/Carlsen.pgn"
max_games = config.max_games
save_model = True

device = torch.device("mps")
files = list_npy_files("data/training") #TODO: Read validation and training sets separately
full_dataset = NPZDataset(files, device, batch_size=100)
# full_dataset = PGNDataset(path, device, max_games=max_games)
train_split_ratio = 0.8
training_size = int(train_split_ratio * len(full_dataset))
validation_size = len(full_dataset) - training_size

training_dataset, validation_dataset = random_split(full_dataset, [training_size, validation_size])

training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=config.batch_size,
    shuffle=True,
)

validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=config.batch_size,
    shuffle=False
)

model = t.ChessModel()
model.to(device)
model.train()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(config.num_epochs):
    print(f"--- Epoch {epoch} ---")
    model.train()
    for i, (input, target) in enumerate(training_dataloader):
        optimizer.zero_grad()
        # Mask for non-padding tokens (padding_idx=0)
        mask = (target != 0)
        output = model(input)
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target = target.view(-1)  # (batch*seq_len,)
        # Only compute loss on non-padding tokens
        if mask.any():
            loss = loss_fn(output[mask], target[mask])
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    for v_i, (val_input, val_target) in enumerate(validation_dataloader):
        with torch.no_grad():
            val_output = model(val_input)
            val_output = val_output.view(-1, val_output.size(-1))
            val_target = val_target.view(-1)
            val_mask = (val_target != 0)
            if val_mask.any():
                val_loss = loss_fn(val_output[val_mask], val_target[val_mask])
    
    # TODO: Running loss?
    print(f"Epoch {epoch}: Loss = {loss.item():.5f} Validation Loss = {val_loss.item():.5f}")

    if epoch % 2 == 0 and save_model:
        torch.save(model.state_dict(), f"models/model-{datetime.now(): %Y-%m-%d-%H-%M-%S}-epoch-{epoch}.pt")