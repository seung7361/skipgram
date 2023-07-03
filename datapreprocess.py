import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from datasets import load_dataset

tokenizer = Tokenizer()
tokenizer.load()

wikitext = load_dataset('wikitext', 'wikitext-103-v1')['train']['text']

dataset = []
for sentence in tqdm(wikitext):
    if len(sentence.split()) < 10:
        continue

    dataset.append(
        tokenizer.tokenize(sentence)
    )

torch.save(dataset, 'dataset.pt')
print(len(dataset))
print(dataset[0], len(dataset[0]))