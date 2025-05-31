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
        embedded = self.embedding(x)

        if embedded.ndim == 3:  # for x_batch (B, L, D)
            # agregate the sequence dimension (L)
            context_representation = embedded.mean(
                dim=1
            )  # (B, D)
            return context_representation
        elif embedded.ndim == 2:  # for y_batch (B, D)
            return embedded
        else:
            raise ValueError(
                f"Unsupported input dimension: {embedded.ndim}. Expected 2 or 3."
            )
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(config.PREDICTOR.output_dim, 512)
        self.l2 = nn.Linear(512, config.VOCAB_SIZE)

    def forward(self, x):
        x = self.l1(x)
        x = nn.GELU()(x)
        x = self.l2(x)
        return x

class TextGenJepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.decoder = Decoder()

    def forward(self, x):
        # x shape: (BATCH_SIZE, BLOCK_SIZE)
        context_embedding = self.encoder(
            x
        )  # Output shape: (BATCH_SIZE, EMBEDDING_DIM)
        
        predicted_embedding = self.predictor(
            context_embedding
        )  # Output shape: (BATCH_SIZE, EMBEDDING_DIM)
        
        decoder_logits = self.decoder(
            predicted_embedding
        )  # Output shape: (BATCH_SIZE, VOCAB_SIZE)
        
        return predicted_embedding, decoder_logits