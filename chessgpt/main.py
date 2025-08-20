import torch
import chessgpt.model.config as config
from torch.utils.data import DataLoader
from chessgpt.model import transformer as t
from chessgpt.model.data import NpyDataset
from chessgpt.pgn_to_npy import list_npy_files
from chessgpt.model.tokenizer import special_tokens_to_embeddings
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence

device = torch.device("mps")

training_files = list_npy_files("data/training")
validation_files = list_npy_files("data/validation")

training_dataset = NpyDataset(training_files, device)
validation_dataset = NpyDataset(validation_files, device) 

def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    # if not input_ids or not target_ids:
    #     torch.zeros(1,1, device=device), torch.zeros(1,1, device=device)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=special_tokens_to_embeddings['<|pad|>'])
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=special_tokens_to_embeddings['<|pad|>'])
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
# TODO: Disabling the scheduler seems to have led to loss dropping more steadily
# TODO: It may have been because I was calling scheduler every batch, but should have been every epoch
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print(f"Training on approximately {len(training_dataset) // config.batch_size} batches.")
print(f"Validating on approximately {len(validation_dataset) // config.batch_size} batches.")
print(f"Batch size: {config.batch_size}")


best_loss = torch.inf 
for epoch in range(config.num_epochs):
    print(f"--- Epoch {epoch} ---")
    model.train()
    total_loss = 0.0
    total_tokens = 0
    val_total_loss = 0.0
    val_total_tokens = 0
    for i, (input, target) in enumerate(training_dataloader):

        optimizer.zero_grad()
        output = model(input)
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target = target.view(-1)  # (batch*seq_len,)

        mask = (target != special_tokens_to_embeddings['<|pad|>'])
        if mask.shape[0] != output.shape[0]:
            print(f"Skipping training batch {i} due to shape mismatch: mask {mask.shape}, output {output.shape}")
            continue
        if mask.any():
            loss = loss_fn(output[mask], target[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            running_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            if i % 10 == 0:
                print(f"Training Epoch: {epoch} | Batch: {i} | Sample input: {input[0][:2]} | Running Loss: {running_loss:.5f} | Running Perplexity: {torch.exp(torch.tensor(running_loss)):.5f}")
        
       # scheduler.step()

    model.eval()
    for v_i, (val_input, val_target) in enumerate(validation_dataloader):
        with torch.no_grad():
            val_output = model(val_input)
            val_output = val_output.view(-1, val_output.size(-1))
            val_target = val_target.view(-1)
            val_mask = (val_target != special_tokens_to_embeddings['<|pad|>'])
            if val_mask.shape[0] != val_output.shape[0]:
                print(f"Skipping validation batch {v_i} due to shape mismatch: mask {val_mask.shape}, output {val_output.shape}")
                continue
            if val_mask.any():
                val_loss = loss_fn(val_output[val_mask], val_target[val_mask])
                val_total_loss += val_loss.item() * val_mask.sum().item()
                val_total_tokens += val_mask.sum().item()
                val_running_loss = val_total_loss / val_total_tokens if val_total_tokens > 0 else float('inf')
                if i % 10 == 0:
                    print(f"Validating Epoch: {epoch} | Batch: {v_i} | Sample input: {val_input[0][:2]} | Running Loss: {val_running_loss:.5f} | Running Perplexity: {torch.exp(torch.tensor(val_running_loss)):.5f}")

    
    train_perplexity = torch.exp(torch.tensor(running_loss))
    print(f"Epoch {epoch}: Avg Loss = {running_loss:.5f} | Avg Perplexity: {train_perplexity:.5f} | Avg Val Loss: {val_running_loss:.5f} | Avg Val Perplexity: {torch.exp(torch.tensor(val_running_loss)):.5f}")

    if running_loss < best_loss:
        print(f"Saving model. Model had less loss than last epoch {running_loss} < {best_loss} | Perplexity: {train_perplexity}")
        torch.save(model.state_dict(), f"models/model-{datetime.now(): %Y-%m-%d-%H-%M-%S}-epoch-{epoch}-perplexity-{train_perplexity:.3f}.pt")
        best_loss = running_loss