from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import fasttext     # type: ignore

from data import *
from modules import *
from utils import train

# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
_char_enc, pos_enc = create_encoders(train_data)

# Load the fastText model for English
ft_model = fasttext.load_model("cc.en.100.bin")

# Encode the train set
enc_train = encode_with(train_data, ft_model, pos_enc)
enc_dev = encode_with(dev_data, ft_model, pos_enc)

# Report size of the training data
print("# Train size =", len(enc_train))
print("# Dev size =", len(enc_dev))

class Joint(nn.Module):
    """Joint POS tagging / dependency parsing module.

    Type: EncInp -> Tuple[PosScores, DepScores]

    where:

    * EncInp is an embedded input sentence (2d tensor)
    * PosScores is a Tensor[N x T] with POS-related scores,
      where T is the number of distinct POS tags.
    * DepScores is a Tensor[N x (N + 1)] of dependency-related scores,
      one score vector of size (N + 1) per word.

    The two components (POS tagging and dependency parsing) are based
    on a common contextualized embedding representation.
    """

    def __init__(self,
        ft_model,                   # fastText model for input words
        pos_enc: Encoder[POS],      # Encoder for POS tags
        emb_size: int,              # Embedding size
        hid_size: int               # Hidden size used in LSTMs
    ):
        super().__init__()

        # Keep encoding objects for future use
        self.ft_model = ft_model
        self.pos_enc = pos_enc

        # Common part of the model: LSTM contextualization
        self.context = SimpleBiLSTM(emb_size, hid_size)

        # Scoring module for the POS tagging task
        self.score_pos = nn.Linear(hid_size, pos_enc.size())

        # Biaffine dependency scoring module
        self.score_dep = Biaffine(hid_size)

    def forward(self, xs: EncInp) -> Tuple[Tensor, Tensor]:
        embs = self.context(xs)
        return (self.score_pos(embs), self.score_dep(embs))

    @torch.no_grad()
    def tag(self, sent: List[Word]) -> List[POS]:
        """Tag a sentence with POS tags."""
        xs = encode_input(sent, self.ft_model)
        embs = self.context(xs)
        scores = self.score_pos(embs)
        ys = torch.argmax(scores, dim=1)
        return [self.pos_enc.decode(y.item()) for y in ys]

    @torch.no_grad()
    def parse(self, sent: List[Word]) -> List[Head]:
        """Predicted a dependency head for each word in a sentence."""
        xs = encode_input(sent, self.ft_model)
        embs = self.context(xs)
        scores = self.score_dep(embs)
        ys = torch.argmax(scores, dim=1)
        return [y.item() for y in ys]

model = Joint(ft_model, pos_enc, emb_size=100, hid_size=200)

def pos_accuracy(model, data: List[Tuple[EncInp, EncOut]]):
    """Calculate the POS tagging accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x)[0], dim=1)
        correct += (y[0] == pred_y).sum()
        total += len(y[0])
    return float(correct) / total

def dep_accuracy(model, data: List[Tuple[EncInp, EncOut]]):
    """Calculate the head prediction accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x)[1], dim=1)
        correct += (y[1] == pred_y).sum()
        total += len(y[1])
    return float(correct) / total

def loss(pred: Tuple[Tensor, Tensor], gold: Tuple[Tensor, Tensor]) -> Tensor:
    criterion = nn.CrossEntropyLoss()
    return criterion(pred[0], gold[0]) + criterion(pred[1], gold[1])

train(model, enc_train, enc_dev, loss, pos_accuracy, epoch_num=10, learning_rate=0.001, report_rate=1)
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.0001, report_rate=1)