import torch
import torch.nn as nn


class Replace(nn.Module):

    """A replacement module which, given a (vector) tensor of integer indices,
    replaces each value in the vector with a certain pre-specified index value `ix`,
    with a certain pre-specified probability `p`.
    Examples:
    
    Create the module with replacement probability 0.5 and the special
    index equal to 5
    >>> frg = Replace(p=0.5, ix=5)
    Check if the module preserves the shape of the input matrix
    >>> x = torch.tensor([0, 1, 0, 3, 4, 2, 3])
    >>> frg(x).shape == x.shape
    True
    When `p` is set to 0, the module should behave as an identity function
    >>> frg = Replace(p=0.0, ix=5)
    >>> (frg(x) == x).all().item()
    True
    When `p` is set to 1, all values in the input tensor should
    be replaced by `ix`
    >>> frg = Replace(p=1.0, ix=5)
    >>> (frg(x) == 5).all().item()
    True
    In the evaluation mode, the module should also behave as an identity,
    whatever the probability `p`
    >>> frg = Replace(p=0.5, ix=5)
    >>> _ = frg.eval()
    >>> (frg(x) == x).all().item()
    True
    Make sure the module is actually non-deterministic and returns
    different results for different applications
    >>> frg = Replace(p=0.5, ix=5)
    >>> x = torch.randint(5, (20,))    # length 20, values in [0, 5)
    >>> results = set()
    >>> for _ in range(1000):
    ...     results.add(frg(x))
    >>> assert len(results) > 100
    See if the number of special index values the resulting tensor
    contains on average is actually close to 0.5 * len(x)
    >>> special_ix_num = [(y == 5).sum().item() for y in results]
    >>> avg_special_ix_num = sum(special_ix_num) / len(special_ix_num)
    >>> 0.5*len(x) - 0.5 <= avg_special_ix_num <= 0.5*len(x) + 0.5
    True
    """

    def __init__(self, p: float, ix: int):
        super().__init__()
        self.repl_ix = ix
        self.p = p

    def forward(self, ixs):
        if self.training:
            assert ixs.dim() == 1
            mask = (torch.empty_like(ixs, dtype=torch.float).uniform_() > self.p).long()
            unmask = 1 - mask    # XOR
            return ixs*mask + self.repl_ix*unmask
        else:
            return ixs


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
