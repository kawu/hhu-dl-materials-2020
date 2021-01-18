import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class Apply(nn.Module):
    """Apply a given pure function or module."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class Map(nn.Module):
    """Apply a given module to each element in the list."""

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, xs):
        return [self.f(x) for x in xs]

class Sum(nn.Module):
    """Perform torch.sum w.r.t the first dimension."""

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.sum(m, dim=0)


class SimpleLSTM(nn.Module):

    '''
    Contextualise the input sequence of embedding vectors using unidirectional
    LSTM.

    Type: Tensor[N x Din] -> Tensor[N x Dout], where
    * `N` is is the length of the input sequence
    * `Din` is the input embedding size
    * `Dout` is the output embedding size

    Example:

    >>> lstm = SimpleLSTM(3, 5)   # input size 3, output size 5
    >>> xs = torch.randn(10, 3)   # input sequence of length 10
    >>> ys = lstm(xs)             # equivalent to: lstm.forward(xs)
    >>> list(ys.shape)
    [10, 5]
    '''

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        # Initial ,,hidden state''
        self.h0 = nn.Parameter(torch.randn(out_size).view(1, -1))
        # Initial ,,cell state''
        self.c0 = nn.Parameter(torch.randn(out_size).view(1, -1))
        # LSTM computation cell
        self.cell = nn.LSTMCell(input_size=inp_size, hidden_size=out_size)

    def forward(self, xs):
        '''Apply the LSTM to the input sequence.

        Arguments:
        * xs: a tensor of shape N x Din, where N is the input sequence length
            and Din is the embedding size

        Output: a tensor of shape N x Dout, where Dout is the output size
        '''
        # Initial hidden and cell states
        h, c = self.h0, self.c0
        # Output: a sequence of tensors
        ys = []
        for x in xs:
            # Compute the new hidden and cell states
            h, c = self.cell(x.view(1, -1), (h, c))
            # Emit the hidden state on output; the cell state will only by
            # used to calculate the subsequent states
            ys.append(h.view(-1))
        return torch.stack(ys)


class SimpleBiLSTM(nn.Module):

    '''Bidirectional LSTM: a combination of a forward and a backward LSTM.

    Type: Tensor[N x Din] -> Tensor[N x Dout], where
    * `N` is is the length of the input sequence
    * `Din` is the input embedding size
    * `Dout` is the output embedding size

    WARNING: the output size is required to be divisible by 2!

    Example:

    >>> lstm = SimpleBiLSTM(3, 6) # input size 3, output size 6
    >>> xs = torch.randn(10, 3)   # input sequence of length 10
    >>> ys = lstm(xs)             # equivalent to: lstm.forward(xs)
    >>> list(ys.shape)
    [10, 6]
    '''

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        assert out_size % 2 == 0, "Output size have to be even"
        self.f = SimpleLSTM(inp_size, out_size // 2)
        self.b = SimpleLSTM(inp_size, out_size // 2)

    def forward(self, xs):
        ys1 = self.f(xs)
        ys2 = reversed(self.b(reversed(xs)))
        return torch.cat((ys1, ys2), dim=-1)