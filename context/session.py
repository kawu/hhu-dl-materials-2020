import torch
import torch.nn as nn

from data import parse_and_extract, create_encoders, encode_with
from modules import *

# Uncomment for reproducibility
torch.manual_seed(0)

# Extract the training set and dev set
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
# Create the encoders for input words and output POS tags
word_enc, pos_enc = create_encoders(train_data)
# Encode the dataset
enc_train = encode_with(train_data, word_enc, pos_enc)
enc_dev = encode_with(dev_data, word_enc, pos_enc)

baseline = nn.Sequential(
    # Replace(p=0.1, ix=word_enc.size()),
    nn.Embedding(word_enc.size()+1, 50, padding_idx=word_enc.size()),
    SimpleBiLSTM(50, 50),
    nn.Linear(50, pos_enc.size())
)

def accuracy(model, data):
    """Calculate the accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        correct += (y == pred_y).sum()
        total += len(y)
    return float(correct) / total

def train(
    model: nn.Module,
    train_data,
    dev_data,
    loss,
    epoch_num=10,
    learning_rate=0.001,
    report_rate=10,
):
    """SGD training function."""
    # Use Adam to adapt the model's parameters
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for k in range(epoch_num):
        # Turn on the training mode
        model.train()
        # Variable to store the total loss on the training set
        total_loss = 0
        # Optional: use random dataset permutation in each epoch
        for i in torch.randperm(len(train_data)):
            x, y = train_data[i]
            z = loss(model(x), y)
            total_loss += z.item()
            z.backward()
            optim.step()
            optim.zero_grad()
        if k == 0 or (k+1) % report_rate == 0:
            with torch.no_grad():
                model.eval()
                train_acc = accuracy(model, train_data)
                dev_acc = accuracy(model, dev_data)
                print(f'@{k+1}: loss(train)={total_loss:.3f}, acc(train)={train_acc:.3f}, acc(dev)={dev_acc:.3f}')

# Report size of the training data
print("# Train size =", len(enc_train))
print("# Dev size =", len(enc_dev))

loss = nn.CrossEntropyLoss()
train(baseline, enc_train, enc_dev, loss, epoch_num=10, learning_rate=0.001, report_rate=1)
# => @1: loss(train)=2135.324, acc(train)=0.775, acc(dev)=0.744
# => @2: loss(train)=1108.957, acc(train)=0.857, acc(dev)=0.800
# => @3: loss(train)=768.808, acc(train)=0.899, acc(dev)=0.825
# => @4: loss(train)=552.067, acc(train)=0.932, acc(dev)=0.846
# => @5: loss(train)=395.307, acc(train)=0.952, acc(dev)=0.857
# => @6: loss(train)=282.418, acc(train)=0.968, acc(dev)=0.866
# => @7: loss(train)=199.932, acc(train)=0.979, acc(dev)=0.872
# => @8: loss(train)=137.422, acc(train)=0.986, acc(dev)=0.871
# => @9: loss(train)=95.575, acc(train)=0.991, acc(dev)=0.880
# => @10: loss(train)=65.386, acc(train)=0.994, acc(dev)=0.878
