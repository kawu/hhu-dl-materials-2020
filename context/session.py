import torch
import torch.nn as nn

# import data
from data import encode, parse_and_extract

# For reproducibility
torch.manual_seed(0)

# Read and encode the training data
train_data = encode(parse_and_extract("dev.conllu"))
enc_data = train_data['data']
word_enc = train_data['word_enc']
pos_enc = train_data['pos_enc']

# Let's recreate the baseline model
baseline = nn.Sequential(
    nn.Embedding(word_enc.size(), 25),
    nn.Linear(25, pos_enc.size())
)

# Use cross entropy loss as the objective function
loss = nn.CrossEntropyLoss()

# Use Adam to adapt the baseline model's parameters
optim = torch.optim.Adagrad(baseline.parameters(), lr=0.001)

# Perform SGD for 1000 epochs
for k in range(1000):
    total_loss: float = 0.0
    for i in torch.randperm(len(enc_data)):
        x, y = enc_data[i]
        z = loss(baseline(x), y)
        total_loss += z.item()
        z.backward()
        optim.step()	# version of `nudge` provided by `Adam`
    if k % 50 == 0:
        print(f'@{k}: {total_loss}')
    total_loss = 0.0

# # Let's verify the final losses
# for x, y in enc_data:
#     print(loss(baseline(x), y))
