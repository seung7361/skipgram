import json
import numpy as np
import torch

from tqdm import tqdm
from datasets import load_dataset

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0

        self.special_tokens = [
            '[UNK]',
            '[SOS]',
            '[EOS]',
            '[PAD]'
        ]
    
    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.vocab_size
            self.reverse_vocab[self.vocab_size] = word
            self.vocab_size += 1
    
    def load(self):
        with open('tokenizer.json', 'r') as f:
            file = json.load(f)
        
        if len(file['vocab']) == 0:
            print('Tokenizer is empty')

            wikitext = load_dataset('wikitext', 'wikitext-103-v1')['train']['text']

            for sentence in tqdm(wikitext):
                for word in sentence.split():
                    word = word.lower()
                    if word not in self.vocab:
                        self.add_word(word)

            print('wikitext dataset preprocessing done successfully')

            file['vocab'] = self.vocab
            file['reverse_vocab'] = self.reverse_vocab

            # save file
            with open('tokenizer.json', 'w') as f:
                json.dump(file, f)
        
        else:
            print('Loading Tokenizer from tokenizer.json...')

            self.vocab = file['vocab']
            self.reverse_vocab = file['reverse_vocab']
            self.vocab_size = len(self.vocab)

            print('load complete')
        
        # process special tokens

        for token in self.special_tokens:
            self.add_word(token)
    
    def word_to_idx(self, word):
        if word == '<unk>':
            return self.vocab['[UNK]']

        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab['[UNK]']
    
    def idx_to_word(self, index):
        index = str(index)
        if index in self.reverse_vocab:
            return int(self.reverse_vocab[index]) - 1
        else:
            return -1

    def tokenize(self, sentence, max_length=128):
        out = torch.LongTensor([
            self.word_to_idx(word) for word in sentence.split()
            if self.word_to_idx(word) != 229463
        ])

        while len(out) < max_length:
            out = torch.cat([out, torch.LongTensor([self.vocab['[PAD]']])])
        
        return out[:max_length]
