import numpy as np
import torch
import deepspeed
deepspeed.init_distributed()

from tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup

tokenizer = Tokenizer()
tokenizer.load()

### hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 3e-3
num_epochs = 5
batch_size = 64
embedding_dim = 512
WINDOW_SIZE = 2
MAX_LENGTH = 128

###

# wikitext = load_dataset('wikitext', 'wikitext-103-v1')['train']['text']
print('preprocessing wikitext data....')
train_dataset = []
# for sentence in tqdm(wikitext):
#     if len(sentence.strip().split()) < 12:
#         continue
#     train_dataset.append(sentence.strip())
# print('preprocessing wikitext data done!')
tinystories = load_dataset('roneneldan/TinyStories')['train']['text']
for sentence in tqdm(tinystories):
    if len(sentence.strip().split()) < 12:
        continue
    train_dataset.append(sentence.strip())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    
    def forward(self, inputs):
        # inputs: (batch_size, seq_len)

        embeddings = self.embedding_layer(inputs)
        # embeddings: (batch_size, seq_len, embedding_dim)

        out = self.linear_layer(embeddings)
        # out: (batch_size, seq_len, num_embeddings) => logit values

        out = torch.nn.functional.log_softmax(out, dim=1)
        
        return out


def train_skipgram(model, train_dataloader, num_epochs, lr=learning_rate, WINDOW_SIZE=WINDOW_SIZE, k=10):
    model.cuda()
    model.train()

    loss_out = []

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=3000, num_training_steps=len(train_dataloader)*5)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} started')

        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Epoch: {epoch + 1}, Loss: 0")
        step = 0

        for item in pbar:
            optimizer.zero_grad()

            batch = len(item)
            tokens = torch.cat(
                [tokenizer.tokenize(item[i]) for i in range(batch)], dim=0
            )
            length = tokens.shape[0]

            center_word = model(torch.LongTensor([tokens[i] for i in range(WINDOW_SIZE, length - WINDOW_SIZE)])).cuda()
            context_words = torch.LongTensor([
                [ tokens[i + e] for e in range(-WINDOW_SIZE, WINDOW_SIZE + 1) if e != 0] for i in range(WINDOW_SIZE, length - WINDOW_SIZE)
            ]).cuda()

            loss = sum(
                loss_fn(center_word, context_words[:, j].contiguous()) for j in range(WINDOW_SIZE * 2)
            )
            
            # negative sampling
            for _ in range(k):
                neg_words = torch.LongTensor(np.random.choice(tokenizer.vocab_size, size=(length - WINDOW_SIZE * 2, WINDOW_SIZE * 2))).cuda()
                loss += sum(
                    loss_fn(center_word, neg_words[:, j].contiguous()) for j in range(WINDOW_SIZE * 2)
                )

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {loss.item() / (WINDOW_SIZE * 2)}")
            
            
            scheduler.step()
            step += 1

            if step % 10000 == 0:
                torch.save(model.state_dict(), f'./checkpoints/model2_epoch{epoch + 1}_step{step}.pt')

        torch.save(model.state_dict(), f'./checkpoints/model_epoch{epoch + 1}.pt')
        print(f"checkpoint for epoch {epoch + 1} was saved.")
    
    return loss_out

model = EmbeddingLayer(tokenizer.vocab_size, embedding_dim).cuda()
# linear = LinearLayer(embedding_dim, tokenizer.vocab_size)
# model = Skipgram(embedding, linear, tokenizer.vocab_size, embedding_dim).cuda()
print("model parameters: {:_}".format(sum(p.numel() for p in model.parameters())))

loss_values = train_skipgram(model, train_dataloader, num_epochs=num_epochs, lr=learning_rate)

torch.save(model.state_dict(), 'model.pt')
torch.save(loss_values, 'loss_values.pt')
print('exit')