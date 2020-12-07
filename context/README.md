# Contextualisation

The code developed during the session will be placed in
[session.py](session.py).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Dataset](#dataset)
- [Baseline](#baseline)
- [Accuracy](#accuracy)
- [Development set](#development-set)
- [OOV words](#oov-words)
- [Contextualisation methods](#contextualisation-methods)
  - [LSTM](#lstm)
- [Footnotes](#footnotes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Dataset

<!--
TODO:
* Increate the size of the dataset (use a fragment of ParTUT?)
* Split it to train/dev/test
* Read data from disk?
-->

<!--
[official    repository][UD_German-HDT-repo] (either use `git` or `Code -> Download ZIP`)
-->

1. Download and unzip the [`UD_English-ParTUT`][UD_English-ParTUT] dataset (v.2.2)
  <!-- <sup>[1](#footnote1)</sup>) from [here][dataset] -->
1. Use [data.py](data.py) to read and encode the dataset
```python
from data import parse_and_extract, create_encoders, encode_with

# Extract the training set
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
# Create the encoders for input words and output POS tags
word_enc, pos_enc = create_encoders(train_data)
# Encode the dataset
enc_train = encode_with(train_data, word_enc, pos_enc)
```


## Baseline

Our baseline model applies [nn.Linear][linear] to directly score the word
embedding representations.
```python
import torch
import torch.nn as nn

# # Uncomment for reproducibility
# torch.manual_seed(0)

baseline = nn.Sequential(
    nn.Embedding(data.word_enc.size(), 10),
    nn.Linear(10, data.tag_enc.size())
)
```

## Accuracy

<!--
TODO:
* Explain and implement accuracy
* Show the result of the baseline model on the (dev->)train set
-->

POS tagging accuracy is defined as the percentage of words for which the
classifier predicts the correct POS tag.  One way to implement it is:
```python
def accuracy(model, data):
    """Calculate the accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        correct += (y == pred_y).long().sum()
        total += len(y)
    return float(correct) / total
```

**Exercise**: Update the [SGD training loop](https://github.com/kawu/hhu-dl-materials-2020/tree/main/gradient-descent#stochastic-gradient-descent)
so that the accuracy of the baseline model is reported every `K` epochs.  Train a model to make sure it works.

**Exercise**: In addition to accuracy on the training set, report the total loss on the training set (every `K` epochs).

## Development set

Currently we only check the loss/accuracy of the model on the training set, but
in practice we want to make sure the model generalises well to data unseen
during training.  To this end, we can:
* Extract the development dataset and encode it using the encoders created on
  the training set (**Q**: Could we just create the encoders on the entire
  training+development dataset instead?)
* Report the accuracy of the model on the develoment set (dev set for short)
  during training, along with the accuracy on the training set

**Exercise**: Extract and encode the dev set from the `dev.conllu` file.  To
make encoding work, you may need to modify the `encode_with` function in
`data.py`, since some words/POS tags present in `dev.conllu` may not be present
in `train.conllu`.

**Exercise**: Report the accuracy on the dev set together with the accuracy on
the train set during training.

## OOV words

Certain words from the dev set do not occur in the train set.  Such words are
called *out-of-vocabulary* (OOV) words.

**Question**: How does the baseline model handle OOV words?

**Exercise**: Improve support for the OOV words in the baseline model.

## Contextualisation methods

### :construction: Work In Progress :construction:

Contextualisation is a technique of transforming input embeddings to
contextualised embeddings: vector representations which capture the context in
which the input words<sup>[1](#footnote1)</sup> occur.  Formally, a
contextualisation module takes on input a sequence of embedding vectors, and
outputs a sequence of contextualised embedding vectors.  The output sequence
has *the same length* as the input sequence, but the *size of word embeddings
can change*.

### LSTM

A popular contextualisation technique is based on [*recurrent neural
networks*][RNN] (RNNs) in general, and [*long short-term memory*][LSTM] (LSTM)
RNNs in particular.
A RNN is a *recurrent* in the sense that it iteratively applies a certain
neural computation over a sequence of input tensors.  It can be implemented
using a specific ,,computation cell'', e.g. LSTM uses
[LSTMCell](https://pytorch.org/docs/1.6.0/generated/torch.nn.LSTMCell.html?highlight=lstm%20cell#torch.nn.LSTMCell):
```python
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
        self.h0 = torch.randn(out_size).view(1, -1)
        # Initial ,,cell state''
        self.c0 = torch.randn(out_size).view(1, -1)
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
```


**Exercise**: Extend the baseline model with the `SimpleLSTM` module and see if
you can obtain better performance on the dev set.

**Exercise**: Imlement a bidirectional LSTM (BiLSTM).  A BiLSTM is a combination of two LSTMs: forward LSTM and backward LSTM.  The forward LSTM is applied from left to right, the backward LSTM: from right to left.  The former captures the left context, the latter -- the right context.  On output of BiLSTM, the respective outputs of both underlying LSTMs are concatenated.
<!--
Have a look at the list of the [nn.LSTM][nn-lstm] module's available
hyper-parameters.  See if it improves the performance of the model.
-->

In practice, you can use a higher-level [nn.LSTM][nn-lstm] module to integrate a (Bi)LSTM in your PyTorch application.  Here's a
`nn.Module` which encapsulates the PyTorch's [nn.LSTM][nn-lstm] module
transforming embeddings of a given input size to contextualised embeddings of a
given output size.
```python
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
	self.lstm = nn.LSTM(input_size=inp_size, hidden_size=out_size)

    def forward(self, xs):
        '''Apply the LSTM to the input sequence.

        Arguments:
        * xs: a tensor of shape N x Din, where N is the input sequence length
            and Din is the embedding size

        Output: a tensor of shape N x Dout, where Dout is the output size
        '''
        # Apply the LSTM, extract the first value of the result tuple only
        out, _ = self.lstm(xs.view(xs.shape[0], 1, xs.shape[1]))
        # Reshape and return
        return out.view(out.shape[0], out.shape[2])
```

**Note**: The `forward` method of the [nn.LSTM][nn-lstm] module takes a
3-dimensional tensor on input.  Its second dimension is the *batch size* and it
allows to process several input sentences in parallel.  In practice, input
sentences have different lengths, and may be easier to process when stored
in a [packed sequence][packed-seq] (which [nn.LSTM][nn-lstm] also accepts on
input).

**Note**: LSTM is relatively slow when applied to a single sentence, due to the
recurrent/sequential nature of the computation it performs, which inhibits
effective parallelisation.

### Convolution

```python
class SimpleConv(nn.Module):
    def __init__(self, inp_size: int, out_size: int, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)

    def forward(self, x):
        x = x.view(1, x.shape[1], x.shape[0])
        padding = (self.kernel_size - 1, 0)
        out = self.conv(F.pad(x, padding))
        out_reshaped = out.view(out.shape[2], out.shape[1])
        return out_reshaped
```

<!--
### Recurent networks

### Transformer
-->

## Footnotes

<a name="footnote1">1</a>: In general, contextualisation can be applied to any sequential input, whatever the nature of the elements stored in the sequence (as long as we can reasonably embed them!)

<!--
<a name="footnote1">1</a>: The sample consists of the first 1000 sentences from
[UD_German-HDT-master/de_hdt-ud-train-a-1.conllu](https://github.com/UniversalDependencies/UD_German-HDT/blob/23f2f1d5ce1621611604c39c9e1069448ec2eb39/de_hdt-ud-train-a-1.conllu).
-->



[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
<!-- [dataset]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi20/data/UD_German-HDT-sample.zip "UD_German-HDT sample dataset" -->
[UD_English-ParTUT]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi20/data/UD_English-ParTUT.zip "UD_English-ParTUT sample dataset"
[UD_German-HDT-repo]: https://github.com/UniversalDependencies/UD_German-HDT
[RNN]: https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Recurrent_neural_network_unfold.svg "RNN"
[LSTM]: https://colah.github.io/posts/2015-08-Understanding-LSTMs "LSTM"
[nn-lstm]: https://pytorch.org/docs/1.6.0/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM "LSTM nn.Module"
[packed-seq]: https://pytorch.org/docs/1.6.0/generated/torch.nn.utils.rnn.PackedSequence.html?highlight=packedsequence#torch.nn.utils.rnn.PackedSequence "Packed sequence"

<!--
[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[module]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Module.html?highlight=module#torch.nn.Module "PyTorch neural module"
[cross-entropy]: https://en.wikipedia.org/wiki/Cross_entropy "Cross entropy"
[cross-entropy-loss]: https://pytorch.org/docs/1.6.0/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss "Cross entropy loss criterion"
[eucl-dist]: https://pytorch.org/docs/1.6.0/generated/torch.dist.html?highlight=dist#torch.dist "Euclidean distance"
[sgd]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method "Stochastic gradient descent"
[sgd-optim]: https://pytorch.org/docs/1.6.0/optim.html?highlight=sgd#torch.optim.SGD "SGD optimiser"
[adam-optim]: https://pytorch.org/docs/1.6.0/optim.html?highlight=adam#torch.optim.Adam "Adam optimiser"
[adagrad-optim]: https://pytorch.org/docs/1.6.0/optim.html#torch.optim.Adagrad "Adagrad optimiser"
-->
