import torch
from tokenizer import Tokenizer
from datasets import load_dataset
from tqdm import tqdm


### hyperparameters

embedding_dim = 512

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


def train(
    train_dataset, tokenizer, model,
    num_epochs=100, learning_rate=1e-3, WINDOW_SIZE=2,
):
    loss_fn = torch.nn.NLLLoss(ignore_index=52985)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    
    for epoch in range(num_epochs):
        print(f'epoch: {epoch}')

        pbar = tqdm(train_dataset)
        pbar.set_description(f'Epoch: {epoch}, Loss: -')

        step = 0

        for sentence in pbar:
            optimizer.zero_grad()
            if len(sentence.split()) < 10:
                continue
            input_ids = tokenizer.tokenize(sentence).cuda()

            center_words = input_ids[WINDOW_SIZE:len(input_ids) - WINDOW_SIZE].contiguous().cuda()
            context_words = torch.LongTensor([
                [
                    *[input_ids[center + j] for j in range(-WINDOW_SIZE, WINDOW_SIZE + 1) if j != 0]
                ] for center in range(WINDOW_SIZE, len(input_ids) - WINDOW_SIZE)
            ]).cuda()
            
            log_probs = model(center_words)
            loss = sum(
                loss_fn(log_probs, context_words[:, i]) for i in range(WINDOW_SIZE * 2)
            )

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                pbar.set_description(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

            step += 1
            # if step % 5000 == 0:
            #     torch.save(model.state_dict(), './checkpoints/model_{}_{}.pt'.format(epoch, step))
    
        torch.save(model.state_dict(), './checkpoints/model_{}.pt'.format(epoch + 1))


tokenizer = Tokenizer()
tokenizer.load()

tinystories = load_dataset('roneneldan/TinyStories')['train']['text'][:200000]
model = Word2Vec(tokenizer.vocab_size, embedding_dim=embedding_dim).cuda()

words = []
for sentence in tinystories:
    words += sentence.split()
words = set(words)
print(words)

train(train_dataset=tinystories, tokenizer=tokenizer, model=model)