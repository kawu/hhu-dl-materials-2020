import torch
# import torch.nn as nn
# import torch.nn.functional as F

from utils import accuracy, train
from data import parse_and_extract, create_encoders, encode_with
from modules import *

# For reproducibility
# torch.manual_seed(0)``

# Read datasets
train_data = parse_and_extract("train.conllu")
dev_data = parse_and_extract("dev.conllu")
print(f'# train = {len(train_data)}')
print(f'# dev = {len(dev_data)}')

# Create encoders
word_enc, pos_enc = create_encoders(train_data)
enc_train = encode_with(train_data, word_enc, pos_enc)
enc_dev = encode_with(dev_data, word_enc, pos_enc)

# emb.weight.requires_grad = False

# Let's create the baseline model
baseline = nn.Sequential(
    Forget(word_enc.size(), p=0.0),
    nn.Embedding(word_enc.size()+1, 100, padding_idx=word_enc.size()),
    # nn.Dropout(p=0.25),
    # SimpleBiLSTM(50, 50),
    # Concat2d(
    #     SimpleBiLSTM(25, 50),
    #     nn.Identity()
    # ),
    SimpleTransformer(100),
    # SimpleConv(24, 24, kernel_size=2),
    # nn.Linear(60, 30),
    # nn.LeakyReLU(),
    nn.Linear(100, pos_enc.size()),
    # nn.Linear(10, pos_enc.size())
)

# Use cross entropy loss as the objective function
loss = nn.CrossEntropyLoss()

# train(baseline, loss, 6, learning_rate=0.0001)
# train(baseline, loss, 4, learning_rate=0.00001)

# train(baseline, loss, enc_train, enc_dev, 6, learning_rate=0.0001, report_rate=1)
# train(baseline, loss, enc_train, enc_dev, 4, learning_rate=0.00001, report_rate=1)

train(baseline, loss, enc_train, enc_dev, 10, learning_rate=0.0001, report_rate=5)
# train(baseline, loss, enc_train, enc_dev, 20, learning_rate=0.00005, report_rate=5)
train(baseline, loss, enc_train, enc_dev, 20, learning_rate=0.00001, report_rate=5)
