from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from data import *
from modules import *
from utils import train

# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
char_enc, pos_enc = create_encoders(train_data)

# Encode the train set
enc_train = encode_with(train_data, char_enc, pos_enc)
enc_dev = encode_with(dev_data, char_enc, pos_enc)

# Report size of the training data
print("# Train size =", len(enc_train))
print("# Dev size =", len(enc_dev))

model = nn.Sequential(
    Map(nn.Sequential(
        nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size()),
        SimpleLSTM(50, 200),
        Apply(lambda xs: xs[-1]),
    )),
    SimpleBiLSTM(200, 200),
    Biaffine(200),
)

def dep_accuracy(model, data: List[Tuple[EncInp, EncOut]]):
    """Calculate the head prediction accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        correct += (y[1] == pred_y).sum()
        total += len(y[1])
    return float(correct) / total

def loss(pred: Tensor, gold: Tuple[Tensor, Tensor]) -> Tensor:
    criterion = nn.CrossEntropyLoss()
    return criterion(pred, gold[1])

train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.001, report_rate=1)
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.0001, report_rate=1)