import os
import requests
import torch.nn as nn
import torch
import torch.nn.functional as F

def load_text() -> str:
    if not os.path.isfile('input.txt'):
        response = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
        with open('input.txt', 'w') as f:
            f.write(response.text)

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

        return text

text = load_text()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]
print(train_data)


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
