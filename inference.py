import numpy as np
import torch
import math

from tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer()
tokenizer.load()

vocab_size = tokenizer.vocab_size

### hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-3
num_epochs = 5
batch_size = 32
embedding_dim = 512
WINDOW_SIZE = 2
MAX_LENGTH = 128

###

class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        log_probs = torch.nn.functional.log_softmax(out, dim=1)

        return log_probs

# model = Skipgram(
#     embedding=EmbeddingLayer(num_embeddings=vocab_size, embedding_dim=embedding_dim),
#     linear=LinearLayer(input_dim=embedding_dim, output_dim=vocab_size),
#     num_embeddings=vocab_size,
#     embedding_dim=embedding_dim
# ).cuda()

model = Word2Vec(vocab_size, embedding_dim).cuda()
model.load_state_dict(torch.load('./checkpoints/model_67.pt'))

cat = model.embeddings(torch.tensor([tokenizer.word_to_idx('cat')]).cuda())
dog = model.embeddings(torch.tensor([tokenizer.word_to_idx('dog')]).cuda())

print(torch.cosine_similarity(cat, dog))

king = model.embeddings(torch.tensor([tokenizer.word_to_idx('king')]).cuda())
male = model.embeddings(torch.tensor([tokenizer.word_to_idx('male')]).cuda())
female = model.embeddings(torch.tensor([tokenizer.word_to_idx('female')]).cuda())
queen = model.embeddings(torch.tensor([tokenizer.word_to_idx('queen')]).cuda())

queen_emb = model.linear(king - male + female)
print(torch.cosine_similarity(queen_emb, queen))