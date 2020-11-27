import torch
import torch.nn as nn

from utils import accuracy
from data import parse_and_extract, create_encoders, encode_with

# For reproducibility
torch.manual_seed(0)

# Read datasets
train_data = parse_and_extract("train.conllu")
dev_data = parse_and_extract("dev.conllu")

# Create encoders
word_enc, pos_enc = create_encoders(train_data)
enc_train = encode_with(train_data, word_enc, pos_enc)
enc_dev = encode_with(dev_data, word_enc, pos_enc)

# train_data = encode(parse_and_extract("dev.conllu"))
# enc_data = train_data['data']
# word_enc = train_data['word_enc']
# pos_enc = train_data['pos_enc']

# Let's recreate the baseline model
baseline = nn.Sequential(
    nn.Embedding(word_enc.size()+1, 25, padding_idx=word_enc.size()),
    nn.Linear(25, pos_enc.size())
)

# Use cross entropy loss as the objective function
loss = nn.CrossEntropyLoss()

# Use Adam to adapt the baseline model's parameters
optim = torch.optim.Adagrad(baseline.parameters(), lr=0.01)

# Perform SGD for 1000 epochs
for k in range(100):
    total_loss: float = 0.0
    for i in torch.randperm(len(enc_train)):
        x, y = enc_train[i]
        z = loss(baseline(x), y)
        total_loss += z.item()
        z.backward()
        optim.step()
    if k % 10 == 0:
        acc_train = accuracy(baseline, enc_train)
        acc_dev = accuracy(baseline, enc_dev)
        print(
            f'@{k}: loss(train)={total_loss:.3f}, '
            f'acc(train)={acc_train:.3f}, '
            f'acc(dev)={acc_dev:.3f}'
        )
    total_loss = 0.0

# # Let's verify the final losses
# for x, y in enc_data:
#     print(loss(baseline(x), y))
