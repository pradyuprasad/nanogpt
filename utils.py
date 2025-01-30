from typing import List
import requests
from model import Config
import os

def load_text() -> str:
    if not os.path.isfile('input.txt'):
        response = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
        with open('input.txt', 'w') as f:
            f.write(response.text)

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

        return text


config = Config()
text = load_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s:str) -> List[int]:
    return [stoi[c] for c in s]

def decode(nums:List[int]) -> str:
    return ''.join(itos[i] for i in nums)
