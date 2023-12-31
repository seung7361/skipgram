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
    
    def process_word(self, word):
        word = word.lower()
        word = word.replace('\'', '').replace('\"', '').replace('?', '').replace('.', '').replace('!', '').replace(',', '').replace(':', '').replace(';', '').replace('(', '').replace(')', '').encode('ascii', 'ignore').decode().strip().replace('/', '-').split('-')

        if len(word[0]) == 1:
            return [''.join(word)]
        
        else:
            return word

        return word

    
    def load(self):
        with open('tokenizer.json', 'r') as f:
            file = json.load(f)
        
        if len(file['vocab']) == 0:
            print('Tokenizer is empty')

            tinystories = load_dataset('roneneldan/TinyStories')['train']['text']

            for sentence in tqdm(tinystories):
                for word in sentence.split():
                    word = self.process_word(word)
                    
                    if len(word) == 1:
                        if word[0] not in self.vocab:
                            self.add_word(word[0])

                    else:
                        if len(word[0]) == 1:
                            attached_word = ''.join(word)
                            if attached_word not in self.vocab:
                                self.add_word(attached_word)
                        else:
                            for each in word:
                                if each not in self.vocab:
                                    self.add_word(each)

            print('tinystories dataset preprocessing done successfully')

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

    def tokenize(self, sentence, max_length=85):
        out = []
        for words in sentence.split():
            for word in self.process_word(words):
                if self.word_to_idx(word) != self.vocab['[UNK]']:
                    out.append(self.word_to_idx(word))
        out = torch.LongTensor(out)

        while len(out) < max_length:
            out = torch.cat([out, torch.LongTensor([self.vocab['[PAD]']])])
        
        return out[:max_length]

