import numpy as np
import torch

from preprocessing import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

tokenizer = Tokenizer()
tokenizer.load()

### hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 2e-4
num_epochs = 10
batch_size = 32

wikitext = load_dataset('wikitext', 'wikitext-103-v1')['train']['text']

train_dataset = torch.load('train_dataset.pt')
# train_dataset = [tokenizer.tokenize(sentence) for sentence in tqdm(wikitext) if len(sentence.split()) > 10]

# torch.save(train_dataset, 'train_dataset.pt')
# print(train_dataset[0])

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

class SkipgramDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size=3):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        length = len(self.data[idx])
        out = []

        for i in range(self.window_size, length - self.window_size):
            center_word = torch.LongTensor([self.data[idx][i]])
            context_words = torch.LongTensor([ self.data[idx][i + offset] for offset in range(-self.window_size, self.window_size + 1) if offset != 0 ])
        
            out.append((center_word, context_words))

        return out


def train_skipgram(model, dataset, num_epochs, lr=learning_rate):
    model.train()

    loss_out = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} started')
        pbar = tqdm(dataset)
        for item in pbar:
            total_loss, cnt = 0.0, 0

            for center_word, context_words in item:
                optimizer.zero_grad()

                log_probs = model(center_word)
                print(log_probs.shape, model(context_words).shape)
                loss = sum(loss_fn(log_probs, cw.unsqueeze(0)) for cw in model(context_words))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1
            
            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {total_loss / cnt}")
            loss_out.append(total_loss)
    
    return loss_out

model = Skipgram(tokenizer.vocab_size, 512)
dataset = SkipgramDataset(train_dataset, window_size=3)

train_skipgram(model, dataset, num_epochs=num_epochs, lr=learning_rate)

torch.save(model.state_dict(), 'model.pt')
print('exit')