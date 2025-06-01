import torch
from safetensors.torch import load_model
import os

from config import config
from models import TextGenJepa
from utils import model_to_device, device


def top_k_filtering(logits, top_k: int):
    if top_k == 0:
        return logits
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def top_p_filtering(logits, top_p: float):
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1
    )
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        ..., :-1
    ].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return torch.where(
        indices_to_remove, torch.full_like(logits, float("-inf")), logits
    )


def generate(
    model: TextGenJepa,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)

    generated_token_ids = list(token_ids)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > config.BLOCK_SIZE:
            current_model_input = input_tensor[:, -config.BLOCK_SIZE :]
        else:
            current_model_input = input_tensor

        if current_model_input.size(1) == 0:
            break

        with torch.no_grad():
            _, next_token_logits = model(
                current_model_input
            )  # Shape: (1, VOCAB_SIZE)

        if temperature <= 0.0:  # Deterministic: argmax
            next_token_id = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )
        else:
            next_token_logits = next_token_logits / temperature
            next_token_logits = top_k_filtering(next_token_logits, top_k)
            next_token_logits = top_p_filtering(next_token_logits, top_p)

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(
                probs, num_samples=1
            )  # Shape: (1,1)

        generated_token_ids.append(next_token_id.item())
        input_tensor = torch.cat((input_tensor, next_token_id), dim=1)

        # Optional: Implement EOS token check if applicable
        # if next_token_id.item() == tokenizer.eos_token_id:
        #     break

    return tokenizer.decode(generated_token_ids)


if __name__ == "__main__":
    model = TextGenJepa()
    checkpoint_path = (
        "checkpoints/gen/best_model_epoch1_batch8400.safetensors"
    )

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print(
            "Please ensure the path is correct and the model is trained."
        )
        exit(1)

    try:
        load_model(model, checkpoint_path)
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        exit(1)

    model = model_to_device(model)
    model.eval()

    tokenizer = config.tokenizer

    prompts_and_settings = [
        {
            "id": "Example 1 (User's first)",
            "prompt": "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:",
            "max_new_tokens": 60,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
        },
        {
            "id": "Example 2 (User's second)",
            "prompt": "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst",
            "max_new_tokens": 60,
            "temperature": 0.75,
            "top_k": 0, # top_k disabled
            "top_p": 0.85,
        },
        {
            "id": "Example 3 (User's third)",
            "prompt": "First ",
            "max_new_tokens": 40,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.0, # top_p disabled
        },
        {
            "id": "Creative Writing Start",
            "prompt": "In a realm of floating islands and whispered prophecies, a young cartographer named Elara discovered a map unlike any other. It depicted not lands, but",
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.92,
        },
        {
            "id": "More deterministic",
            "prompt": "The old clock tower struck midnight. In the silence that followed,",
            "max_new_tokens": 50,
            "temperature": 0.5, # Lower temperature
            "top_k": 10,
            "top_p": 0.9,
        },
         {
            "id": "Argmax (temp <= 0)",
            "prompt": "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:",
            "max_new_tokens": 60,
            "temperature": 0.0, # Should trigger argmax
            "top_k": 0,
            "top_p": 0.0,
        },
    ]

    for ps in prompts_and_settings:
        print(f"--- Generating for: {ps['id']} ---")
        print(f"Prompt:\n{ps['prompt']}")
        generated_text = generate(
            model,
            tokenizer,
            ps["prompt"],
            ps["max_new_tokens"],
            temperature=ps.get("temperature", 1.0),
            top_k=ps.get("top_k", 0),
            top_p=ps.get("top_p", 0.9),
        )
        print(
            f"\nGenerated (temp={ps.get('temperature', 1.0)}, k={ps.get('top_k', 0)}, p={ps.get('top_p', 0.9)}):"
        )
        print(generated_text)
        print("\n" + "=" * 70 + "\n")