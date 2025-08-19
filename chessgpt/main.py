import torch
import chessgpt.model.config as config
from torch.utils.data import DataLoader
from chessgpt.model import transformer as t
from chessgpt.model.data import NpyDataset
from chessgpt.pgn_to_npy import list_npy_files
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence

save_model = False
device = torch.device("mps")

training_files = list_npy_files("data/training")
validation_files = list_npy_files("data/validation")

training_dataset = NpyDataset(training_files, device)
validation_dataset = NpyDataset(validation_files, device) 

def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
    return input_ids, target_ids

# TODO: Adjust code to allow for workers outside the main thread (num_workers > 0)
training_dataloader = DataLoader(
    dataset=training_dataset,
    batch_size=config.batch_size,
    collate_fn=collate_fn,
    shuffle=True
)

validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=config.batch_size,
    collate_fn=collate_fn,
    shuffle=False
)

model = t.ChessModel()
model.to(device)
model.train()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print(f"Training on approximately {len(training_dataset) // config.batch_size} batches.")
print(f"Validating on approximately {len(validation_dataset) // config.batch_size} batches.")
print(f"Batch size: {config.batch_size}")

for epoch in range(config.num_epochs):
    print(f"--- Epoch {epoch} ---")
    model.train()
    for i, (input, target) in enumerate(training_dataloader):
        if i % 1 == 0:
            print(f"Epoch: {epoch} | Batch: {i} | Sample: {input[0][:2]}, {target[0][:2]}")
        optimizer.zero_grad()
        output = model(input)
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target = target.view(-1)  # (batch*seq_len,)

        # Mask for non-padding tokens (padding_idx=0)
        mask = (target != 0)
        if mask.shape[0] != output.shape[0]:
            print(f"Skipping training batch {i} due to shape mismatch: mask {mask.shape}, output {output.shape}")
            continue
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
            if val_mask.shape[0] != val_output.shape[0]:
                print(f"Skipping validation batch {i} due to shape mismatch: mask {mask.shape}, output {output.shape}")
                continue
            if val_mask.any():
                val_loss = loss_fn(val_output[val_mask], val_target[val_mask])
    
    # TODO: Running loss?
    print(f"Epoch {epoch}: Loss = {loss.item():.5f} Validation Loss = {val_loss.item():.5f}")

    if epoch % 2 == 0 and save_model:
        torch.save(model.state_dict(), f"models/model-{datetime.now(): %Y-%m-%d-%H-%M-%S}-epoch-{epoch}.pt")