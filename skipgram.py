import numpy as np
import torch

from tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

tokenizer = Tokenizer()
tokenizer.load()

### hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-3
num_epochs = 5
batch_size = 32


train_dataset = torch.load('train_dataset.pt')

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


def train_skipgram(model, dataset, num_epochs, lr=learning_rate):
    model.cuda()
    model.train()

    loss_out = []

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} started')
        step = 0
        pbar = tqdm(dataset)
        pbar.set_description(f"Epoch: {epoch + 1}, Loss: 0")
        for sentence in pbar:
            total_loss, cnt = 0.0, 0

            window_size = 2
            data = tokenizer.tokenize(sentence).to(device)
            length = len(data)

            if length < 10:
                continue

            for i in range(window_size, length - window_size):
                center_word = torch.LongTensor([ data[i] ]).cuda()
                context_words = torch.LongTensor([ data[i + offset] for offset in range(-window_size, window_size + 1) if offset != 0 ]).cuda()

                log_probs = model(center_word)
                context_words = model(context_words)

                loss = sum(loss_fn(log_probs[0], cw) for cw in context_words)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {total_loss / cnt}")
            loss_out.append(total_loss)
        
            step += 1
            if step % 10000 == 0:
                print('checkpoint for epoch {} and step {} was saved.'.format(epoch + 1, step))
                torch.save(model.state_dict(), 'checkpoints/checkpoint_epoch{}_step{}.pt'.format(epoch + 1, step))


        torch.save(model.state_dict(), 'model.pt')
        print(f"checkpoint for epoch {epoch + 1} was saved.")
    
    return loss_out

model = Skipgram(tokenizer.vocab_size, 512).cuda()
print("model parameters: {:_}".format(sum(p.numel() for p in model.parameters())))
print("train data: {:_}", len(wikitext))

loss_values = train_skipgram(model, wikitext, num_epochs=num_epochs, lr=learning_rate)

torch.save(model.state_dict(), 'model.pt')
torch.save(loss_values, 'loss_values.pt')
print('exit')