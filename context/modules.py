import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=inp_size, hidden_size=out_size, bidirectional=False)

    def forward(self, x):
        '''Apply the LSTM to the input sentence.

        Arguments:
        * x: a tensor of shape N x D, where N is the input sentence length
            and D is the embedding size

        Output: a tensor of shape N x O, where O is the output size (a parameter
            of the SimpleLSTM component)
        '''
        # Apply the LSTM, extract the first value of the result tuple only
        out, _ = self.lstm(x.view(x.shape[0], 1, x.shape[1]))
        # Reshape and return
        return out.view(out.shape[0], out.shape[2])


class SimpleBiLSTM(nn.Module):

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        # Make sure out_size is an even number
        if not out_size % 2 == 0:
            raise RuntimeError(f'Provided out size = {out_size} must be even')
        self.lstm = nn.LSTM(
            input_size=inp_size, hidden_size=out_size//2,
            bidirectional=True, num_layers=1, dropout=0.0,
        )

    def forward(self, x):
        '''Apply the LSTM to the input sentence.

        Arguments:
        * x: a tensor of shape N x D, where N is the input sentence length
            and D is the embedding size

        Output: a tensor of shape N x O, where O is the output size (a parameter
            of the SimpleLSTM component)
        '''
        # Apply the LSTM, extract the first value of the result tuple only
        out, _ = self.lstm(x.view(x.shape[0], 1, x.shape[1]))
        # Reshape and return
        return out.view(out.shape[0], out.shape[2])


class SimpleTransformer(nn.Module):

    def __init__(self, dim_size: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_size, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        # print(f'x: {x.shape}')
        out = self.transformer_encoder(x.view(x.shape[0], 1, x.shape[1]))
        # print(f'out: {out.shape}')
        return out.view(out.shape[0], out.shape[2])


# class SimpleConv(nn.Module):

#     def __init__(self, inp_size: int, out_size: int, kernel_size=1):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.conv = nn.Conv1d(
#             # inp_size, out_size, kernel_size=1+ctx_size*2, padding=ctx_size)
#             inp_size, out_size,
#             kernel_size=kernel_size,
#             padding=kernel_size // 2
#         )

#     def forward(self, x):
#         print(f'x: {x.shape}')
#         out = self.conv(x.view(1, x.shape[1], x.shape[0]))
#         out_reshaped = out.view(out.shape[2], out.shape[1])
#         print(f'out: {out_reshaped.shape}')
#         if self.kernel_size % 2 == 0:
#             return out_reshaped[:-1]
#         else:
#             return out_reshaped


class SimpleConv(nn.Module):
    def __init__(self, inp_size: int, out_size: int, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)
        # nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = x.view(1, x.shape[1], x.shape[0])
        padding = (self.kernel_size - 1, 0)
        out = self.conv(F.pad(x, padding))
        out_reshaped = out.view(out.shape[2], out.shape[1])
        return out_reshaped


class Concat2d(nn.Module):

    def __init__(self, m1: nn.Module, m2: nn.Module):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        assert x.dim() == 2
        y1 = self.m1(x)
        y2 = self.m2(x)
        assert y1.dim() == y2.dim() == 2
        assert y1.shape[0] == y2.shape[0]
        y = torch.cat([y1, y2], dim=1)
        assert y.shape[1] == y1.shape[1] + y2.shape[1]
        return y


class Forget(nn.Module):

    def __init__(self, repl_ix: int, p=0.1):
        super().__init__()
        self.repl_ix = repl_ix
        self.p = p

    def forward(self, ixs):
        if self.training:
            assert ixs.dim() == 1
            mask = (torch.empty_like(ixs, dtype=torch.float).uniform_() > self.p).long()
            # mask = (torch.randn(ixs.shape[0]) > 0).long()
            unmask = 1 - mask    # XOR
            return ixs*mask + self.repl_ix*unmask
        else:
            return ixs
