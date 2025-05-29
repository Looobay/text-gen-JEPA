import torch.nn as nn
import torch
from config import config

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(config.PREDICTOR.input_dim, config.PREDICTOR.hidden_dim)
        self.l2 = nn.Linear(config.PREDICTOR.hidden_dim, config.PREDICTOR.output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = nn.GELU()(x)
        x = self.l2(x)
        return x

class Encoder(nn.Module):
    # TODO: the Encoder should be a small Transformer to encode the context and micro details
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=0)
        return x

class TextGenJepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

class KTokenPredictor(nn.Module):
    """ Predict the amount of K tokens in the predictor's output vector"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(config.PREDICTOR.output_dim, 200)
        self.l2 = nn.Linear(200, 1)
    
    def forward(self, x):
        n = self.l1(x)
        n = nn.GELU()(n)
        n = self.l2(n)
        n = torch.round(n)
        return n

class LogitsGenerator(nn.Module):
    """ Generate the logits for the sequence of tokens """
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(config.PREDICTOR.output_dim, config.VOCAB_SIZE)
    
    def forward(self, x):
        x = self.l1(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_token_predictor = KTokenPredictor()
        self.logits_generator = LogitsGenerator()

    def forward(self, x):
        k_tokens = self.k_token_predictor(x)
        k = int(k_tokens.item())
        logits = self.logits_generator(x)
        _, top_k_indices = torch.topk(logits, k, dim=1)
        return top_k_indices