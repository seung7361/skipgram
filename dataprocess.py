import numpy as np
import torch

from tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

# wikitext = load_dataset('wikitext', 'wikitext-103-v1')['train']['text']

tokenizer = Tokenizer()
tokenizer.load()

data = torch.load('dataset.pt')
train_dataset = []

### hyperparamters

WINDOW_SIZE = 2

###

for text in tqdm(data):
    for i in range(WINDOW_SIZE, len(text) - WINDOW_SIZE):
        train_dataset.append(torch.LongTensor([
            text[i],
            *[text[i + e] for e in range(-WINDOW_SIZE, WINDOW_SIZE + 1) if e != 0]
        ]))

torch.save(train_dataset, 'train_dataset.pt')

print(len(train_dataset))
print(train_dataset[0])
print(train_dataset[0].shape)
print(train_dataset[1].shape)
print(train_dataset[2].shape)