from data import parse_and_extract, create_encoders, encode_with

import torch
import torch.nn as nn

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
    # TODO: Make sure `padding_idx` is necessary
    nn.Embedding(word_enc.size()+1, 10, padding_idx=word_enc.size()),
    nn.Linear(10, pos_enc.size())
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
    # Use Adam to adapt the baseline model's parameters
    optim = torch.optim.Adam(baseline.parameters(), lr=learning_rate)
    for k in range(epoch_num):
        # Variable to store the total loss on the training set
        total_loss = 0
        # Optional: use random dataset permutation in each epoch
        for i in torch.randperm(len(train_data)):
            x, y = train_data[i]
            z = loss(baseline(x), y)
            total_loss += z.item()
            z.backward()
            optim.step()
        if k == 0 or (k+1) % report_rate == 0:
            # TODO: enter the evaluation mode
            # TODO: consider switching off gradient calculation
            train_acc = accuracy(model, train_data)
            dev_acc = accuracy(model, dev_data)
            print(f'@{k+1}: loss(train)={total_loss:.3f}, acc(train)={train_acc:.3f}, acc(dev)={dev_acc:.3f}')

# Report size of the training data
print("# Train size =", len(enc_train))

loss = nn.CrossEntropyLoss()
train(baseline, enc_train, enc_dev, loss, epoch_num=50, learning_rate=0.00005, report_rate=5)
