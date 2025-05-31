import tiktoken
from utils import device
embed_dim = 768

class PredictorConfig:
    def __init__(self):
        self.input_dim = embed_dim
        self.hidden_dim = 2048
        self.output_dim = embed_dim

class Config:
    def __init__(self):
        self.tokenizer_model = "gpt-2"
        self.tokenizer = tiktoken.encoding_for_model(self.tokenizer_model)
        self.TRAIN_SPLIT = 0.8 # REMINDER: if you change this data please reset the dataset you are using!
        self.PREDICTOR = PredictorConfig()
        self.VOCAB_SIZE = self.tokenizer.n_vocab
        self.EMBEDDING_DIM = embed_dim
        self.LR = 1e-4
        self.EPOCHS = 1
        self.device = device
        self.BATCH_SIZE = 20 # seq per batch
        self.BLOCK_SIZE = 64 # seq len
        self.EMA_DECAY = 0.99 # EMA decay rate
        self.EVAL_INTERVAL = 25  # evaluate every N batchs
        self.DECODER_LOSS_WEIGHT = 0.5 # weight of the decoder loss

    def print(self):
        print("===CONFIG===")
        print(f"Vocab size: {config.VOCAB_SIZE}")
        print(f"Tokenizer from: {config.tokenizer_model}")
        print(f"Embedding dim: {config.EMBEDDING_DIM}")
        print(f"Train split: {config.TRAIN_SPLIT}")
        print(f"Device: {config.device}")
        print(f"Epochs: {config.EPOCHS}")
        print(f"Predictor hidden dim: {config.PREDICTOR.hidden_dim}")
        print(f"Predictor output dim: {config.PREDICTOR.output_dim}")
        print(f"Predictor input dim: {config.PREDICTOR.input_dim}")
        print(f"EMA decay: {config.EMA_DECAY}")
        print(f"Eval interval: {config.EVAL_INTERVAL}")
        print(f"Learning rate: {config.LR:.2e}")
        print(f"Decoder loss weight: {config.DECODER_LOSS_WEIGHT}")
        print("============")

config = Config()