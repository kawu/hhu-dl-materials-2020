from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from bert_serving.client import BertClient  # type: ignore

from data import *
from modules import *
from utils import train

# Configuration constants
DEVICE = 'cuda'

# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
_char_enc, pos_enc = create_encoders(train_data)

# Create BERT client
bert_client = BertClient()

# Encode the train set
enc_train = encode_with(train_data, bert_client, pos_enc, device=DEVICE)
enc_dev = encode_with(dev_data, bert_client, pos_enc, device=DEVICE)

# Report size of the training data
print("# Train size =", len(enc_train))
print("# Dev size =", len(enc_dev))

class Joint(nn.Module):
    """Joint POS tagging / dependency parsing module.

    Type: List[EncInp] -> Tuple[PosScores, DepScores]

    where:

    * EncInp is an embedded input sentence (2d tensor)
    * PosScores is a Tensor[B x N x T] with POS-related scores,
      where T is the number of distinct POS tags and B is the
      size of the batch and N is the maximum sentence length
      in the batch.
    * DepScores is a Tensor[B x N x (N + 1)] of dependency-related scores,
      one score vector of size (N + 1) per word.

    The two components (POS tagging and dependency parsing) are based
    on a common contextualized embedding representation.
    """

    def __init__(self,
        bert_client,                # BERT client
        pos_enc: Encoder[POS],      # Encoder for POS tags
        emb_size: int,              # Embedding size
        hid_size: int,              # Hidden size used in LSTMs
        device                      # Device to put the model parameters on
    ):
        super().__init__()

        # Keep encoding objects for future use
        self.bert_client = bert_client
        self.pos_enc = pos_enc

        # Common part of the model: LSTM contextualization
        self.context = BiLSTM(emb_size, hid_size)

        # Scoring module for the POS tagging task
        self.score_pos = nn.Linear(hid_size, pos_enc.size())

        # Biaffine dependency scoring module
        self.score_dep = Biaffine(hid_size)

        # Dummy parameter to retrieve the device
        self.dummy_param = nn.Parameter(torch.empty(0))

        # Move the module to the target device
        self.to(device)

    def forward(self, xs: List[EncInp]) -> Tuple[Tensor, Tensor]:
        # Create a tensor with sentence lengths
        ns = torch.tensor([len(x) for x in xs])
        # Convert the input list to a packed sequence
        seq = rnn.pack_sequence(xs, enforce_sorted=False)
        # Contextualize all sentences in the sequence
        ctx = self.context(seq)
        # Convert the sequence to a packed representation
        embs, _ns = rnn.pad_packed_sequence(ctx, batch_first=True)
        # Check that the sentence length match, just in case
        assert (ns == _ns).all()
        # Calculate and return the scores
        pos_scores = self.score_pos(embs)
        dep_scores = self.score_dep(embs)
        return (pos_scores, dep_scores)

    def forward1(self, xs: EncInp) -> Tuple[Tensor, Tensor]:
        pos_scores, dep_scores = self.forward([xs])
        assert pos_scores.shape[0] == 1
        assert dep_scores.shape[0] == 1
        return (pos_scores[0], dep_scores[0])

    @torch.no_grad()
    def tag(self, sent: List[Word]) -> List[POS]:
        """Tag a sentence with POS tags."""
        device = self.dummy_param.device
        xs = encode_input(sent, self.bert_client, device=device)
        embs = self.context.forward1(xs)
        scores = self.score_pos(embs)
        ys = torch.argmax(scores, dim=1)
        return [self.pos_enc.decode(y.item()) for y in ys]

    @torch.no_grad()
    def parse(self, sent: List[Word]) -> List[Head]:
        """Predicted a dependency head for each word in a sentence."""
        device = self.dummy_param.device
        xs = encode_input(sent, self.bert_client, device=device)
        embs = self.context.forward1(xs)
        scores = self.score_dep.forward1(embs)
        ys = torch.argmax(scores, dim=1)
        return [y.item() for y in ys]

model = Joint(bert_client, pos_enc, emb_size=128, hid_size=200, device=DEVICE)

def pos_accuracy(model, data: List[Tuple[EncInp, EncOut]]):
    """Calculate the POS tagging accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model.forward1(x)[0], dim=1)
        correct += (y[0] == pred_y).sum()
        total += len(y[0])
    return float(correct) / total

def dep_accuracy(model, data: List[Tuple[EncInp, EncOut]]):
    """Calculate the head prediction accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model.forward1(x)[1], dim=1)
        correct += (y[1] == pred_y).sum()
        total += len(y[1])
    return float(correct) / total

# def batch_loss(model: Joint, batch: List[Tuple[EncInp, EncOut]]):
#     '''Cumulative cross-entropy loss of the model on the given dataset.

#     Parameters
#     ----------
#     model : Joint
#         Joint POS tagging / dependency parsing model
#     batch : List[Tuple[EncInp, EncOut]]
#         List of int-encoded input/output pairs
#     '''
#     # Use cross-entropy loss as training criterion
#     criterion = nn.CrossEntropyLoss(reduction='sum')
#     # Create lists of input sentences and target values, respectively
#     xs = [x for x, _y in batch]
#     ys = [y for _x, y in batch]
#     # Sentence length in the batch
#     ns = list(map(len, xs))
#     # Apply the model to the input
#     pos_scores_batch, dep_scores_batch = model(xs)
#     # Calculate the loss values sentence by sentence
#     batch_loss = 0.0
#     for i in range(len(xs)):
#         # Length of the i-th sentence
#         n = ns[i]
#         # POS scores and dependency scores for the i-th sentence
#         pos_scores = pos_scores_batch[i][:n]
#         dep_scores = dep_scores_batch[i][:n][:n+1]
#         # Targets POS tags and dependencies
#         pos_gold, dep_gold = ys[i]
#         # Apply the criterion and update the cumulative batch loss
#         batch_loss += criterion(pos_scores, pos_gold) + \
#             criterion(dep_scores, dep_gold)
#     return batch_loss

def batch_loss(model: Joint, batch: List[Tuple[EncInp, EncOut]]):
    '''Cumulative cross-entropy loss of the model on the given dataset.

    Parameters
    ----------
    model : Joint
        Joint POS tagging / dependency parsing model
    batch : List[Tuple[EncInp, EncOut]]
        List of int-encoded input/output pairs
    '''
    # Use cross-entropy loss as training criterion
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    # Create lists of input sentences and target values, respectively
    xs = [x for x, _y in batch]
    ys = [y for _x, y in batch]
    # Gold POS tags and dependencies, padded
    pos_gold = pad([pos for pos, _dep in ys], padding_value=-1)
    dep_gold = pad([dep for _pos, dep in ys], padding_value=-1)
    # Apply the model to the input
    pos_scores, dep_scores = model(xs)
    # Calculate the POS-related cumulative loss
    pos_loss = criterion(pos_scores.permute(0, 2, 1), pos_gold)
    dep_loss = criterion(dep_scores.permute(0, 2, 1), dep_gold)
    return pos_loss + dep_loss

def pad(xs: List[Tensor], padding_value) -> Tensor:
    '''Pad and stack a list of tensors.

    Parameters
    ----------
    xs : List[Tensor]
        List of tensors with the same number of dimensions and the same dtype.
        Their sizes along the first dimension can differ, but the remaining
        dimensions must be the same.
    padding_value
        Value used for padding (of the same dtype as the tensors in `xs`)

    Examples
    --------
    >>> x = torch.tensor([0, 1, 2])
    >>> y = torch.tensor([3, 4, 5, 6])
    >>> z = torch.tensor([7])
    >>> pad([x, y, z], padding_value=-1)
    tensor([[ 0,  1,  2, -1],
            [ 3,  4,  5,  6],
            [ 7, -1, -1, -1]])
    >>> xs = torch.tensor([[0, 1], [3, 4]])
    >>> ys = torch.tensor([[5, 6], [7, 8], [9, 10]])
    >>> pad([xs, ys], padding_value=-1)[0]
    tensor([[ 0,  1],
            [ 3,  4],
            [-1, -1]])
    >>> pad([xs, ys], padding_value=-1)[1]
    tensor([[ 5,  6],
            [ 7,  8],
            [ 9, 10]])

    The size of the input tensors can differ only along the first dimension:
    >>> zs = torch.tensor([[5, 6, 7]])
    >>> pad([xs, zs], padding_value=-1) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    RuntimeError: ...
    '''
    seq = rnn.pack_sequence(xs, enforce_sorted=False)
    ys, _ns = rnn.pad_packed_sequence(seq,
        batch_first=True, padding_value=padding_value)
    return ys

train(model, enc_train, enc_dev, batch_loss, pos_accuracy,
    epoch_num=10, learning_rate=0.001, report_rate=5, batch_size=16)
train(model, enc_train, enc_dev, batch_loss, dep_accuracy,
    epoch_num=10, learning_rate=0.0001, report_rate=5, batch_size=16)
