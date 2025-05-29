import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from utils import model_to_device, print_parameters
from models import TextGenJepa, Encoder, Decoder
from config import config
from safetensors.torch import save_file

config.print()

# Check for dataset argument
if len(sys.argv) < 2:
    print("Error 👾: Dataset name not provided. Usage: python train.py <dataset_name> --decoder (optional)")
    exit(1)
elif len(sys.argv) > 2 and sys.argv[2] == "--decoder":
    print("⚠️ WARNING: You are ONLY training the decoder!")
    print("TODO: Implement the training loop for the decoder")
    dec = model_to_device(Decoder())
    print(f"Decoder Parameters: {print_parameters(dec)}")
    optimizer = torch.optim.AdamW(dec.parameters(), lr=config.LR)
    exit(0)

# Seting up data
path = f"data/{sys.argv[1]}"
train_bin_path = path+"/train.bin"
val_bin_path = path+"/val.bin"

for path in [train_bin_path, val_bin_path]:
    if os.path.getsize(path) == 0:
        print(f"Error 👾: File {path} is empty")
        exit(1)
    else:
        print(f"😄 File {path} found")

# loading tokens
train_data = np.fromfile(train_bin_path, dtype=np.uint16)
val_data = np.fromfile(val_bin_path, dtype=np.uint16)

dead_zones = {"train": set(), "val": set()} # already used tokens

def get_batch(training: bool, block_size, batch_size):
    """
    split: 'train' or 'val'
    block_size: sequence length
    batch_size: number of sequences
    eval: if True, refresh dead_zones
    """
    global dead_zones
    data = train_data if training else val_data
    data_len = len(data)
    max_start = data_len - block_size - 1

    split = "train" if training else "val"

    if max_start < 0:
        raise ValueError(f"block_size ({block_size}) is too large for the dataset size ({data_len}).")

    if not training:
        dead_zones[split] = set()  # refresh at each eval
    
    available_indices = data_len - (block_size + 1) # Number of possible start indices
    if len(dead_zones[split]) >= available_indices and split == "train": # Only for training, eval can see all data
        print(f"Warning: All data from '{split}' split has been seen in this epoch. Resetting dead_zones for this split.")
        dead_zones[split] = set()

    x_list, y_list, used = [], [], 0
    attempts = 0 # Safety break for the while loop
    max_attempts = (max_start + 1) * 2 # Allow for some collisions

    while used < batch_size:
        if attempts > max_attempts:
            print(f"Warning: Could not find enough unique samples for split '{split}' after {max_attempts} attempts. Batch may be smaller than requested or data might be exhausted for this epoch.")
            if not x_list: # If no samples could be found at all
                 # Decide how to handle: maybe return None, or an empty tensor, or raise error
                 # For now, let's return what we have, even if it's an empty batch
                 # Or, if you expect full batches, you could raise an error or reset dead_zones earlier.
                 # If training, and dead_zones covers all possible indices, reset it.
                if len(dead_zones[split]) >= available_indices and split == "train":
                    print(f"Resetting dead_zones for split '{split}' as all data has been seen.")
                    dead_zones[split] = set()
                    attempts = 0 # Reset attempts to try again
                    continue # Try to fill the batch again
                else: # If not all data seen, or it's eval split, and still can't get a batch
                    print(f"Returning a potentially incomplete batch for split '{split}'.")
                    break # Break and return whatever was collected
            else:
                break # Return the partially filled batch

        idx = np.random.randint(0, max_start + 1)
        if idx in dead_zones[split]:
            attempts += 1
            continue
        # Move tensors to the correct device
        x = torch.tensor(data[idx:idx+block_size], dtype=torch.long).to(config.device)
        y = torch.tensor(data[idx+1:idx+block_size+1], dtype=torch.long).to(config.device)
        x_list.append(x)
        y_list.append(y)
        dead_zones[split].add(idx)
        used += 1
        attempts = 0 # Reset attempts after a successful addition

    if not x_list: # If batch is empty after the loop (e.g. max_attempts reached with no success)
        # This can happen if block_size is too large for the remaining data or all data is in dead_zones
        # and not reset.
        # Return empty tensors on the correct device to avoid issues downstream
        # Or handle as an error condition if your training loop expects full batches.
        print(f"Warning: Returning empty batch for split '{split}'. Check dataset size and block_size.")
        return torch.empty((0, block_size), dtype=torch.long, device=config.device), torch.empty((0, block_size), dtype=torch.long, device=config.device)

    x_batch = torch.stack(x_list)
    y_batch = torch.stack(y_list)
    return x_batch, y_batch

# Add evaluation function for validation
def evaluate(model, encoder, block_size, batch_size):
    """
    Evaluate model over a configurable number of validation batches.
    """
    model.eval()
    encoder.eval()
    loss_func = torch.nn.MSELoss()
    eval_losses = []
    
    # Determine the number of evaluation steps/batches.
    # Uses config.EVAL_ITERS if available, otherwise a default value (e.g., 100).
    # Note: Due to get_batch resetting dead_zones['val'] each time, 
    # these batches might sample overlapping data if num_eval_iters * batch_size > len(val_data).
    # For a true evaluation over the entire unique validation set without repetition, 
    # get_batch behavior for 'val' or the evaluation batching strategy would need adjustment.
    num_eval_iters = getattr(config, 'EVAL_ITERS', 100) # Default to 100 iterations

    for i in range(num_eval_iters):
        x_batch, y_batch = get_batch(False, block_size, batch_size)
        
        if x_batch.numel() == 0 or y_batch.numel() == 0:
            if not eval_losses: # No batches processed at all
                print(f"Warning: Could not retrieve any validation batches after {i} attempts. Validation set might be too small or empty.")
                return float('nan') 
            # If some batches were processed, just stop and return average of what was collected
            print(f"Warning: Retrieved an empty batch during validation after {i} iterations. Proceeding with {len(eval_losses)} collected losses.")
            break 
        
        with torch.no_grad():
            hat_s_y_batch = model(x_batch)
            s_y_batch = encoder(y_batch)
            loss = loss_func(hat_s_y_batch, s_y_batch)
            eval_losses.append(loss.item())
            
    if not eval_losses:
        # This case implies num_eval_iters might be 0 or first batch was empty.
        print("Warning: No losses recorded during evaluation. num_eval_iters might be zero or val data is insufficient.")
        return float('nan') 

    return sum(eval_losses) / len(eval_losses)

# Importing models
gen = model_to_device(TextGenJepa())
enc = model_to_device(Encoder())
#dec = model_to_device(Decoder())

print(f"TextGenJepa Parameters: {print_parameters(gen)}")
print(f"Encoder Parameters: {print_parameters(enc)}")

# Optimizer
optimizer = torch.optim.AdamW(gen.parameters(), lr=config.LR)

# Prepare for saving best model
best_val_loss = float('inf')
os.makedirs("checkpoints", exist_ok=True)
print("💾 Checkpoints directory ensured at ./checkpoints/")

progress_bar = tqdm(range(config.EPOCHS), desc="Training", unit="epoch")
loss_store = []
val_loss_store = []
for epoch in progress_bar:
    loss_func = torch.nn.MSELoss()
    gen.train(True)
    # reset seen indices for full coverage each epoch
    dead_zones['train'] = set()
    available_indices = len(train_data) - (config.BLOCK_SIZE + 1)
    num_batches = max(1, available_indices // config.BATCH_SIZE)
    # track batch-level progress
    batch_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", unit="batch", leave=False)
    for batch_idx in batch_bar:
        optimizer.zero_grad()
        x_batch, y_batch = get_batch(True, config.BLOCK_SIZE, config.BATCH_SIZE)
        if x_batch.numel() == 0 or y_batch.numel() == 0:
            continue
        hat_s_y_batch = gen(x_batch)
        s_y_batch = enc(y_batch)
        loss = loss_func(hat_s_y_batch, s_y_batch)
        loss.backward()
        optimizer.step()
        with torch.no_grad():  # EMA update
            for param_online, param_target in zip(gen.encoder.parameters(), enc.parameters()):
                param_target.data = param_target.data * config.EMA_DECAY + param_online.data * (1. - config.EMA_DECAY)
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        loss_store.append({"loss": loss.item(), "epoch": epoch})
        # Evaluate every N batches
        if (batch_idx + 1) % config.EVAL_INTERVAL == 0:
            eval_loss = evaluate(gen, enc, config.BLOCK_SIZE, config.BATCH_SIZE)
            if eval_loss is not None and not np.isnan(eval_loss):
                # Note: original print used epoch+1, consider if batch_idx is more relevant here
                print(f"Eval Epoch {epoch+1}, Batch {batch_idx+1}, Validation Loss: {eval_loss:.4f}") 
                val_loss_store.append({"val_loss": eval_loss, "epoch": epoch+1, "batch": batch_idx+1}) # also storing batch_idx
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    print(f"🎉 New best validation loss: {best_val_loss:.4f}. Saving model weights...")
                    # Consider appending batch_idx to filenames if saving frequently
                    save_file(gen.state_dict(), f"checkpoints/gen/best_epoch{epoch+1}_batch{batch_idx+1}.safetensors")
                    save_file(enc.state_dict(), f"checkpoints/enc/best_epoch{epoch+1}_batch{batch_idx+1}.safetensors")
            elif np.isnan(eval_loss):
                 print(f"Eval Epoch {epoch+1}, Batch {batch_idx+1}, Validation Loss: NaN (Skipping)")

if loss_store:
    min_loss_entry = min(loss_store, key=lambda d: d["loss"])
    print(f"Lowest TRAINING loss: {min_loss_entry['loss']:.4f} at epoch {min_loss_entry['epoch']}")
    if val_loss_store:
        best_val = min(val_loss_store, key=lambda d: d['val_loss'])
        print(f"Best VALIDATION Loss: {best_val['val_loss']:.4f} at epoch {best_val['epoch']}")