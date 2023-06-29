import numpy as np
import torch

from preprocessing import Tokenizer

tokenizer = Tokenizer()
tokenizer.load()

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
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(num_embeddings, embedding_dim)
        self.linear_layer = LinearLayer(embedding_dim, num_embeddings)
    
    def forward(self, inputs):
        # inputs: (batch_size, seq_len)

        embeddings = self.embedding_layer(inputs)
        # embeddings: (batch_size, seq_len, embedding_dim)

        out = self.linear_layer(embeddings)
        # out: (batch_size, seq_len, num_embeddings) => logit values
        
        out = torch.nn.functional.softmax(out, dim=1)
        # out: (batch_size, seq_len, num_embeddings) => probability values

        return out

def train_skipgram(model, num_epochs: int, data, lr: float):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss()

    for epoch in range(num_epochs):
        total_loss, cnt = 0.0, 0

        for context in data:
            input = context.split()
            if len(input) < 10:
                continue
            
            context = torch.LongTensor([tokenizer.word_to_idx(word) for word in input])
            target = context

            optimizer.zero_grad()

            prob = model(context, target)

            loss = loss_fn(prob, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss / cnt}")

model = Skipgram(tokenizer.vocab_size, 2048)
print("{:_}".format(sum(p.numel() for p in model.parameters())))