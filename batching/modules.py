import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn

class Biaffine(nn.Module):
    '''Calculate pairwise matching scores.

    Type: Tensor[B x N x D] -> Tensor[B x N x (N + 1)]

        where

            * B is the batch size
            * N is the maximum sentence length in the batch

    For a given sequence (matrix) of word embeddings, calculate the matrix of
    pairwise matching scores.

    Example
    -------

    Sample input (randomized):
    >>> ns = torch.tensor([4, 5, 2])        # Sentence lengths
    >>> B = len(ns)                         # Batch size
    >>> D = 10                              # Embedding size
    >>> bia = Biaffine(D)                   # Biaffine module
    >>> xs = torch.randn(B, max(ns), D)     # Sample embeddings

    Make sure that forward1 and forward give the same results
    for all inputs in the batch:
    >>> ys = bia.forward(xs)
    >>> for i in range(B):
    ...    y1 = bia.forward1(xs[i][:ns[i]])
    ...    y2 = ys[i][:ns[i],:ns[i]+1]
    ...    assert torch.isclose(y1, y2, atol=1e-6).all()
    '''

    def __init__(self, emb_size: int):
        super().__init__()
        # NOTE: Parameter/components are the same as in the
        # non-batching-enabled variant
        self.depr = nn.Linear(emb_size, emb_size)
        self.hedr = nn.Linear(emb_size, emb_size)
        self.root = nn.Parameter(torch.randn(emb_size))

    def forward1(self, xs: Tensor):
        deps = self.depr(xs)
        heds = torch.cat((self.root.unsqueeze(0), self.hedr(xs)))
        return deps @ heds.t()  # torch.mm(deps, heds.t())

    def forward(self, xs: Tensor):
        B = xs.shape[0]    # Batch size
        N = xs.shape[1]    # Maximum sentence length
        D = xs.shape[2]    # Input embedding size

        # Calulate the head/dependent representations
        deps = self.depr(xs)
        root = self.root.view(1, 1, -1).expand(B, 1, -1)
        heds = torch.cat((root, self.hedr(xs)), dim=1)

        # Check the shape (optional)
        assert list(heds.shape) == [B, 1 + N, D]

        # Calculate and return the head scores
        return torch.bmm(deps, heds.permute(0, 2, 1))

# class Apply(nn.Module):
#     """Apply a given pure function or module."""

#     def __init__(self, f):
#         super().__init__()
#         self.f = f

#     def forward(self, x):
#         return self.f(x)

# class Map(nn.Module):
#     """Apply a given module to each element in the list."""

#     def __init__(self, f: nn.Module):
#         super().__init__()
#         self.f = f

#     def forward(self, xs):
#         return [self.f(x) for x in xs]

# class Sum(nn.Module):
#     """Perform torch.sum w.r.t the first dimension."""

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return torch.sum(m, dim=0)


# class SimpleLSTM(nn.Module):

#     '''
#     Contextualise the input sequence of embedding vectors using unidirectional
#     LSTM.

#     Type: Tensor[N x Din] -> Tensor[N x Dout], where
#     * `N` is is the length of the input sequence
#     * `Din` is the input embedding size
#     * `Dout` is the output embedding size

#     Example:

#     >>> lstm = SimpleLSTM(3, 5)   # input size 3, output size 5
#     >>> xs = torch.randn(10, 3)   # input sequence of length 10
#     >>> ys = lstm(xs)             # equivalent to: lstm.forward(xs)
#     >>> list(ys.shape)
#     [10, 5]
#     '''

#     def __init__(self, inp_size: int, out_size: int):
#         super().__init__()
#         # Initial ,,hidden state''
#         self.h0 = nn.Parameter(torch.randn(out_size).view(1, -1))
#         # Initial ,,cell state''
#         self.c0 = nn.Parameter(torch.randn(out_size).view(1, -1))
#         # LSTM computation cell
#         self.cell = nn.LSTMCell(input_size=inp_size, hidden_size=out_size)

#     def forward(self, xs):
#         '''Apply the LSTM to the input sequence.

#         Arguments:
#         * xs: a tensor of shape N x Din, where N is the input sequence length
#             and Din is the embedding size

#         Output: a tensor of shape N x Dout, where Dout is the output size
#         '''
#         # Initial hidden and cell states
#         h, c = self.h0, self.c0
#         # Output: a sequence of tensors
#         ys = []
#         for x in xs:
#             # Compute the new hidden and cell states
#             h, c = self.cell(x.view(1, -1), (h, c))
#             # Emit the hidden state on output; the cell state will only by
#             # used to calculate the subsequent states
#             ys.append(h.view(-1))
#         return torch.stack(ys)


# class SimpleBiLSTM(nn.Module):

#     '''Bidirectional LSTM: a combination of a forward and a backward LSTM.

#     Type: Tensor[N x Din] -> Tensor[N x Dout], where
#     * `N` is is the length of the input sequence
#     * `Din` is the input embedding size
#     * `Dout` is the output embedding size

#     WARNING: the output size is required to be divisible by 2!

#     Example:

#     >>> lstm = SimpleBiLSTM(3, 6) # input size 3, output size 6
#     >>> xs = torch.randn(10, 3)   # input sequence of length 10
#     >>> ys = lstm(xs)             # equivalent to: lstm.forward(xs)
#     >>> list(ys.shape)
#     [10, 6]
#     '''

#     def __init__(self, inp_size: int, out_size: int):
#         super().__init__()
#         assert out_size % 2 == 0, "Output size have to be even"
#         self.f = SimpleLSTM(inp_size, out_size // 2)
#         self.b = SimpleLSTM(inp_size, out_size // 2)

#     def forward(self, xs):
#         ys1 = self.f(xs)
#         ys2 = reversed(self.b(reversed(xs)))
#         return torch.cat((ys1, ys2), dim=-1)


class BiLSTM(nn.Module):
    '''Contextualise the input sequence of embedding vectors using
    bidirectional LSTM.

    Type: PackedSequence -> PackedSequence

    Example
    -------

    Sample input (randomized):
    >>> ns = torch.tensor([4, 5, 2])        # Sentence lengths
    >>> Din = 10                            # Input embedding size
    >>> Dout = 20                           # Output embedding size
    >>> lstm = BiLSTM(Din, Dout)            # BiLSTM module
    >>> xs = [                              # Sample sentences
    ...     torch.randn(n, Din)
    ...     for n in ns
    ... ]
    >>> seq = rnn.pack_sequence(xs,         # Create packed sequence
    ...             enforce_sorted=False)

    Make sure that forward1 and forward give the same results
    for all inputs in the batch:
    >>> out_seq = lstm.forward(seq)
    >>> ys, _ns = rnn.pad_packed_sequence(out_seq, batch_first=True)
    >>> assert (ns == _ns).all()
    >>> for i in range(len(xs)):
    ...    y1 = lstm.forward1(xs[i])
    ...    y2 = ys[i][:ns[i]]
    ...    assert torch.isclose(y1, y2, atol=1e-6).all()
    '''

    def __init__(self, inp_size: int, out_size: int, **kwargs):
        super().__init__()
        assert out_size % 2 == 0, "Output size have to be even"
        self.lstm = nn.LSTM(
            input_size=inp_size,
            hidden_size=out_size // 2,
            bidirectional=True,
            **kwargs    # Other keyword arguments, if any
        )

    def forward1(self, xs: Tensor) -> Tensor:
        out, _ = self.lstm(xs.view(xs.shape[0], 1, xs.shape[1]))
        return out.view(out.shape[0], out.shape[2])

    def forward(self, seq: rnn.PackedSequence) -> rnn.PackedSequence:
        return self.lstm(seq)[0]