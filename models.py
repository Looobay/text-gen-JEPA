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
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch
        self.layers = 2
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        self.rnn = nn.RNN(config.EMBEDDING_DIM, config.EMBEDDING_DIM, self.layers, batch_first=batch)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        if embedded.dim() == 2:
            if self.batch:
                embedded = embedded.unsqueeze(1)
            else:
                embedded = embedded.unsqueeze(0)
        elif embedded.dim() == 3:
            if not self.batch:
                embedded = embedded.transpose(0, 1)
        h0 = torch.zeros(self.layers, batch_size, config.EMBEDDING_DIM).to(x.device)
        _, hn = self.rnn(embedded, h0)
        return hn[-1]

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