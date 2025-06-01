from safetensors.torch import load_model
from models import TextGenJepa
from utils import model_to_device, device
from config import config
import torch

gen = TextGenJepa()
load_model(gen, "checkpoints/gen/best_model_epoch1_batch8400.safetensors")
gen = model_to_device(gen)
gen.eval()

def predict(prompt):
    with torch.no_grad():
        prompt = config.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(prompt, device=device).unsqueeze(0)  # Add batch dim
        
        _, decoder_logits = gen(prompt_tensor)
        
        next_token = torch.argmax(decoder_logits, dim=-1).squeeze().cpu().item()
        
        result = config.tokenizer.decode([next_token])
        return result

def generate(prompt, max_length=30):
    for _ in range(max_length):
        result = predict(prompt)
        prompt += result
    return prompt

print(generate("""First """))