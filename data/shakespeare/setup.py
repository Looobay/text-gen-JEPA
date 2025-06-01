import sys
import os

# project root -> import config.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import requests
import tiktoken
import numpy as np
from config import config

print("Setting up 'tiny_shakespeare' dataset...")

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    print("File not found, downloading...")
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
train_data = data[:int(len(data)*config.TRAIN_SPLIT)]
val_data = data[int(len(data)*config.TRAIN_SPLIT):]

enc = tiktoken.encoding_for_model("gpt-2")
train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)

print(f"Train has {len(train_ids)} tokens")
print(f"Val has {len(val_ids)} tokens")
print(f"Total tokens: {len(train_ids) + len(val_ids)}")

# export tokens to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))