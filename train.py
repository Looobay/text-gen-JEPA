import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from utils import model_to_device, print_parameters
from models import TextGenJepa, Encoder
from config import config
from safetensors.torch import save_file
import glob

config.print()

# Check for dataset argument
if len(sys.argv) < 2:
    print("Error 👾: Dataset name not provided. Usage: python train.py <dataset_name>")
    exit(1)

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

def get_batch(training: bool, block_size, batch_size, device):
    global dead_zones
    data = train_data if training else val_data
    data_len = len(data)
    max_start_idx = data_len - block_size - 1
    split_name = "train" if training else "val"

    if max_start_idx < 0:
        raise ValueError(
            f"block_size ({block_size}) is too large for the dataset size ({data_len})."
        )

    if not training:
        dead_zones[split_name] = set()

    available_indices = data_len - (block_size + 1)
    if (
        len(dead_zones[split_name]) >= available_indices
        and split_name == "train"
    ):
        print(
            f"Warning: All data from '{split_name}' split has been seen in this epoch. Resetting dead_zones."
        )
        dead_zones[split_name] = set()

    x_list, y_list = [], []
    current_batch_size = 0
    attempts = 0
    max_attempts = (max_start_idx + 1) * 2

    while current_batch_size < batch_size:
        if attempts > max_attempts:
            print(
                f"Warning: Could not find enough unique samples for split '{split_name}' after {max_attempts} attempts."
            )
            if not x_list:
                if (
                    len(dead_zones[split_name]) >= available_indices
                    and split_name == "train"
                ):
                    print(
                        f"Resetting dead_zones for '{split_name}' as all data has been seen."
                    )
                    dead_zones[split_name] = set()
                    attempts = 0
                    continue
                else:
                    print(
                        f"Returning a potentially incomplete batch for '{split_name}'."
                    )
                    break
            else:
                break

        idx = np.random.randint(0, max_start_idx + 1)
        if idx in dead_zones[split_name]:
            attempts += 1
            continue

        x = torch.from_numpy(
            data[idx : idx + block_size].astype(np.int64)
        ).to(device)
        y = torch.from_numpy(
            data[idx + 1 : idx + block_size + 1].astype(np.int64)
        ).to(
            device
        )  # y are the targets for x

        # For JEPA, y_target_for_predictor is the next token after the x sequence
        y_target_for_predictor_token_id = torch.tensor(
            [data[idx + block_size]], dtype=torch.long
        ).to(device)

        x_list.append(x)
        y_list.append(y_target_for_predictor_token_id) # Store the single next token ID
        dead_zones[split_name].add(idx)
        current_batch_size += 1
        attempts = 0

    if not x_list:
        return torch.empty(
            (0, block_size), dtype=torch.long, device=device
        ), torch.empty((0,), dtype=torch.long, device=device)

    x_batch = torch.stack(x_list)
    y_batch_next_token_ids = torch.cat(y_list) # Concatenate to get (B,)

    return x_batch, y_batch_next_token_ids

# Add evaluation function for validation
def evaluate(model, target_encoder, mse_loss_fn, ce_loss_fn, device):
    model.eval()
    target_encoder.eval()
    val_losses_jepa = []
    val_losses_decode = []
    val_accuracies = []
    num_eval_iters = getattr(config, "EVAL_ITERS", 100)

    for _ in range(num_eval_iters):
        x_batch, y_next_token_ids_batch = get_batch(
            False, config.BLOCK_SIZE, config.BATCH_SIZE, device
        )

        if x_batch.numel() == 0 or y_next_token_ids_batch.numel() == 0:
            if not val_losses_jepa:
                print("Warning: Could not retrieve any validation batches.")
                return None
            break

        with torch.no_grad():
            predicted_embedding, decoder_logits = model(x_batch)
            target_next_token_embedding = target_encoder(
                y_next_token_ids_batch
            )

            loss_jepa = mse_loss_fn(
                predicted_embedding, target_next_token_embedding
            )
            loss_decode = ce_loss_fn(decoder_logits, y_next_token_ids_batch)
            accuracy = (
                (decoder_logits.argmax(dim=-1) == y_next_token_ids_batch)
                .float()
                .mean()
            )

            val_losses_jepa.append(loss_jepa.item())
            val_losses_decode.append(loss_decode.item())
            val_accuracies.append(accuracy.item())

    if not val_losses_jepa:
        return {
            "val_loss_jepa": float("nan"),
            "val_loss_decode": float("nan"),
            "val_accuracy": float("nan"),
        }

    return {
        "val_loss_jepa": sum(val_losses_jepa) / len(val_losses_jepa),
        "val_loss_decode": sum(val_losses_decode) / len(val_losses_decode),
        "val_accuracy": sum(val_accuracies) / len(val_accuracies),
    }

gen_model = model_to_device(TextGenJepa())
target_encoder_model = model_to_device(Encoder())

print(f"TextGenJepa Parameters: {print_parameters(gen_model)}")
print(f"Target Encoder Parameters: {print_parameters(target_encoder_model)}")

optimizer = torch.optim.AdamW(gen_model.parameters(), lr=config.LR)
mse_loss_func = torch.nn.MSELoss()
ce_loss_func = torch.nn.CrossEntropyLoss()

best_val_total_loss = float("inf")
os.makedirs("checkpoints/gen", exist_ok=True)
os.makedirs("checkpoints/enc", exist_ok=True)
print("💾 Checkpoints directories ensured at ./checkpoints/")

train_loss_store = []
val_metrics_store = []

def cleanup_old_checkpoints(checkpoint_dir, max_files=2):
    """
    Remove oldest checkpoint files, keeping only the max_files most recent ones.
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all .safetensors files in the directory
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
    
    if len(checkpoint_files) <= max_files:
        return
    
    # Sort files by modification time (oldest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Remove oldest files, keeping only max_files
    files_to_remove = checkpoint_files[:-max_files]
    
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"🗑️ Removed old checkpoint: {os.path.basename(file_path)}")
        except OSError as e:
            print(f"Warning: Could not remove {file_path}: {e}")

for epoch in range(config.EPOCHS):
    gen_model.train(True)
    target_encoder_model.eval() 
    dead_zones["train"] = set()
    
    num_train_samples = len(train_data) - (config.BLOCK_SIZE + 1)
    num_batches_per_epoch = max(1, num_train_samples // config.BATCH_SIZE)

    epoch_progress_bar = tqdm(
        range(num_batches_per_epoch),
        desc=f"Epoch {epoch+1}/{config.EPOCHS}",
        unit="batch",
    )

    for batch_idx in epoch_progress_bar:
        optimizer.zero_grad()
        x_batch, y_next_token_ids_batch = get_batch(
            True, config.BLOCK_SIZE, config.BATCH_SIZE, config.device
        )

        if x_batch.numel() == 0 or y_next_token_ids_batch.numel() == 0:
            continue

        predicted_embedding, decoder_logits = gen_model(x_batch)

        with torch.no_grad():
            target_next_token_embedding = target_encoder_model(
                y_next_token_ids_batch
            ).detach()

        loss_jepa = mse_loss_func(
            predicted_embedding, target_next_token_embedding
        )
        loss_decode = ce_loss_func(decoder_logits, y_next_token_ids_batch)
        total_loss = loss_jepa + config.DECODER_LOSS_WEIGHT * loss_decode

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for online_param, target_param in zip(
                gen_model.encoder.parameters(),
                target_encoder_model.parameters(),
            ):
                target_param.data = target_param.data * config.EMA_DECAY + \
                    online_param.data * (1.0 - config.EMA_DECAY)

        epoch_progress_bar.set_postfix(
            total_loss=f"{total_loss.item():.4f}",
            jepa=f"{loss_jepa.item():.4f}",
            decode=f"{loss_decode.item():.4f}",
        )
        train_loss_store.append(
            {
                "total_loss": total_loss.item(),
                "loss_jepa": loss_jepa.item(),
                "loss_decode": loss_decode.item(),
                "epoch": epoch + 1,
                "batch": batch_idx +1
            }
        )

        if (batch_idx + 1) % config.EVAL_INTERVAL == 0:
            eval_metrics = evaluate(
                gen_model, target_encoder_model, mse_loss_func, ce_loss_func, config.device
            )
            if eval_metrics and not np.isnan(eval_metrics["val_loss_decode"]):
                print(
                    f"\nEval Epoch {epoch+1}, Batch {batch_idx+1}: "
                    f"JEPA Loss: {eval_metrics['val_loss_jepa']:.4f}, "
                    f"Decode Loss: {eval_metrics['val_loss_decode']:.4f}, "
                    f"Accuracy: {eval_metrics['val_accuracy']*100:.4f}"
                )
                current_val_metric = eval_metrics["val_loss_decode"]
                eval_metrics["epoch"] = epoch + 1
                eval_metrics["batch"] = batch_idx + 1
                val_metrics_store.append(eval_metrics)

                current_val_total_loss = eval_metrics["val_loss_jepa"] + config.DECODER_LOSS_WEIGHT * eval_metrics["val_loss_decode"]
                
                if current_val_total_loss < best_val_total_loss:
                    best_val_total_loss = current_val_total_loss
                    print(
                        f"🎉 New best weighted validation loss: {best_val_total_loss:.4f}. Saving model..."
                    )
                    save_file(
                        gen_model.state_dict(),
                        f"checkpoints/gen/best_model_epoch{epoch+1}_batch{batch_idx+1}.safetensors",
                    )
                    save_file(
                        target_encoder_model.state_dict(),
                        f"checkpoints/enc/best_model_epoch{epoch+1}_batch{batch_idx+1}.safetensors",
                    )
                    
                    # Clean up old checkpoints to keep only 2 most recent files
                    cleanup_old_checkpoints("checkpoints/gen", max_files=4)
                    cleanup_old_checkpoints("checkpoints/enc", max_files=4)
            elif eval_metrics and np.isnan(eval_metrics["val_loss_decode"]):
                print(
                    f"\nEval Epoch {epoch+1}, Batch {batch_idx+1}: Validation metrics contain NaN. Skipping."
                )

if train_loss_store:
    min_train_loss_entry = min(
        train_loss_store, key=lambda x: x["total_loss"]
    )
    print(
        f"\nLowest TRAINING total loss: {min_train_loss_entry['total_loss']:.4f} "
        f"at epoch {min_train_loss_entry['epoch']}, batch {min_train_loss_entry['batch']}"
    )

if val_metrics_store:
    best_val_decode_entry = min(
        val_metrics_store, key=lambda x: x["val_loss_decode"]
    )
    print(
        f"Best VALIDATION Decode Loss: {best_val_decode_entry['val_loss_decode']:.4f} "
        f"at epoch {best_val_decode_entry['epoch']}, batch {best_val_decode_entry['batch']}"
    )
    best_val_jepa_entry = min(
        val_metrics_store, key=lambda x: x["val_loss_jepa"]
    )
    print(
        f"Best VALIDATION JEPA Loss: {best_val_jepa_entry['val_loss_jepa']:.4f} "
        f"at epoch {best_val_jepa_entry['epoch']}, batch {best_val_jepa_entry['batch']}"
    )
    best_val_acc_entry = max(
        val_metrics_store, key=lambda x: x["val_accuracy"]
    )
    print(
        f"Best VALIDATION Accuracy: {best_val_acc_entry['val_accuracy']:.4f} "
        f"at epoch {best_val_acc_entry['epoch']}, batch {best_val_acc_entry['batch']}"
    )

print("Training finished.")