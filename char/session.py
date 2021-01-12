# PyTorch modules
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from data import create_encoders, encode_with, parse_and_extract
from modules import *
from utils import train

# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
char_enc, pos_enc = create_encoders(train_data)

# Encode the train set
enc_train = encode_with(train_data, char_enc, pos_enc)
enc_dev = encode_with(dev_data, char_enc, pos_enc)

model = nn.Sequential(
    Map(nn.Sequential(
        nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size()),
        SimpleLSTM(50, 200),
        Apply(lambda xs: xs[-1]),
    )),
    SimpleBiLSTM(200, 200),
    nn.Linear(200, pos_enc.size())
)

# # Create the POS tagging model, based on character-level embeddings
# model = nn.Sequential(
#     Map(nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size())),
#     Map(SimpleLSTM(50, 200)),
#     Map(Apply(lambda xs: xs[-1])),
#     SimpleBiLSTM(200, 200),
#     nn.Linear(200, pos_enc.size())
# )

def accuracy(model, data):
    """Calculate the accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        correct += (y == pred_y).sum()
        total += len(y)
    return float(correct) / total

# Report size of the training data
print("# Train size =", len(enc_train))
print("# Dev size =", len(enc_dev))

loss = nn.CrossEntropyLoss()
train(model, enc_train, enc_dev, loss, accuracy, epoch_num=10, learning_rate=0.00005, report_rate=1)

# Inspect the first input sentence
# for word, enc_word in zip(train_data[0][0], enc_train[0][0]):
#     print(word, "=>", enc_word)

# xs = [enc_word for enc_word in enc_train[0][0]]