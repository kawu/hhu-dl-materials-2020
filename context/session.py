import torch
# import torch.nn as nn
# import torch.nn.functional as F

from utils import accuracy
from data import parse_and_extract, create_encoders, encode_with
from modules import *

# For reproducibility
torch.manual_seed(0)

# Read datasets
train_data = parse_and_extract("train.conllu")
dev_data = parse_and_extract("dev.conllu")

# Create encoders
word_enc, pos_enc = create_encoders(train_data)
enc_train = encode_with(train_data, word_enc, pos_enc)
enc_dev = encode_with(dev_data, word_enc, pos_enc)


# Let's recreate the baseline model
baseline = nn.Sequential(
    Forget(word_enc.size(), p=0.1),
    nn.Embedding(word_enc.size()+1, 20, padding_idx=word_enc.size()),
    SimpleBiLSTM(20, 20),
    # SimpleTransformer(24),
    # SimpleConv(24, 24, kernel_size=2),
    nn.Linear(20, pos_enc.size())
)

# Use cross entropy loss as the objective function
loss = nn.CrossEntropyLoss()

# Use Adam to adapt the baseline model's parameters
optim = torch.optim.Adagrad(baseline.parameters(), lr=0.01)

# Perform SGD for 1000 epochs
for k in range(100):
    # Put the model in the training mode at the beginning of each epoch
    baseline.train()
    total_loss: float = 0.0
    for i in torch.randperm(len(enc_train)):
        x, y = enc_train[i]
        z = loss(baseline(x), y)
        total_loss += z.item()
        z.backward()
        optim.step()
    if k == 0 or (k+1) % 10 == 0:
        # Switch off gradient evaluation
        with torch.no_grad():
            baseline.eval() # Put the model in the evaluation mode
            acc_train = accuracy(baseline, enc_train)
            acc_dev = accuracy(baseline, enc_dev)
            print(
                f'@{k+1}: loss(train)={total_loss:.3f}, '
                f'acc(train)={acc_train:.3f}, '
                f'acc(dev)={acc_dev:.3f}'
        )
