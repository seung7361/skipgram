import torch

test = torch.load('train_dataset.pt')
print(len(test))
print(test[0].shape)