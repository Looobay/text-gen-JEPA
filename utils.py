import torch

global device
device = None

def check_backend():
    global device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.cpu.is_available():
        device = torch.device("cpu")

check_backend()

def model_to_device(model):
    model.to(device)
    return model

def print_parameters(model):
    return sum(p.numel() for p in model.parameters())