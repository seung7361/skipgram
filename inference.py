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

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        # num_embeddings == vocab_size
        # embedding_dim == depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weights = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    
    def forward(self, inputs):
        inputs = inputs.long()

        embeddings = self.weights[inputs]

        return embeddings

class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))
    
    def forward(self, inputs):
        outputs = torch.matmul(inputs, self.weights) + self.bias

        return outputs

class Skipgram(torch.nn.Module):
    def __init__(self, embedding, linear, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.embedding_layer = embedding
        self.linear_layer = linear
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inputs):
        # inputs: (batch_size, seq_len)

        embeddings = self.embedding_layer(inputs)
        # embeddings: (batch_size, seq_len, embedding_dim)

        out = self.linear_layer(embeddings)
        # out: (batch_size, seq_len, num_embeddings) => logit values
        
        out = self.softmax(out)
        # out: (batch_size, seq_len, num_embeddings) => probability values

        return out

model = Skipgram(
    embedding=EmbeddingLayer(num_embeddings=vocab_size, embedding_dim=embedding_dim),
    linear=LinearLayer(input_dim=embedding_dim, output_dim=vocab_size),
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
).cuda()

model.load_state_dict(torch.load('./checkpoints/model_epoch5_step20000.pt'))

def get_similarity(word1, word2):
    word1 = model(torch.LongTensor([tokenizer.word_to_idx(word1)])).squeeze()
    word2 = model(torch.LongTensor([tokenizer.word_to_idx(word2)])).squeeze()

    dist = 0.0

    for i in range(len(word1)):
        dist += (word1[i] - word2[i])

    return abs(dist)

print(get_similarity('man', 'woman'))
print(get_similarity('king', 'queen'))
print(get_similarity())