# Contextualization

### :construction: Work In Progress :construction:

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Dataset](#dataset)
- [Baseline](#baseline)
- [Accuracy](#accuracy)
- [Contextualisation](#contextualisation)
  - [Convolution](#convolution)
  - [Recurent networks](#recurent-networks)
  - [Transformer](#transformer)
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

TODO: update the link to data (will use English data after all).

1. Download and unzip a 1000-sentences sample UD dataset (a fragment of
   [`UD_German-HDT`][UD_German-HDT-repo]<sup>[1](#footnote1)</sup>) from
   [here][dataset]
1. The [data.py](data.py) module allows to read and encode the downloaded
   dataset

```python
TODO: Parsed, extract, and examine the training data set.
```


## Baseline

Our baseline model applies [nn.Linear][linear] to directly score the word
embedding representations.
```python
import torch
import torch.nn as nn

import data
from data import enc_data

# For reproducibility
torch.manual_seed(0)

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

### Development set

Currently we only check the loss/accuracy of the model on the training set, but
in practice we want to make sure the model generalises well to data unseen
during training.  To this end, we can:
* Extract the development dataset and encode it using the encoders created on
  the training set (**Q**: could we just create the encoders on the entire,
  training+development dataset instead?)
* Report the accuracy of the model on the develoment set (dev set for short)
  during training, along the accuracy on the training set

**Exercise**: Extract and encode the dev set from the `dev.conllu` file.  To
make encoding work, you may need to modify the `encode_with` function in
`data.py`, since some words/POS tags preset in `dev.conllu` may not be present
in `train.conllu`.

**Exercise**: Report the accuracy on the dev set together with the accuracy on
the train set during training.

## OOV words

Certain words from the dev set do not occur in the train set.  Such words are
called *out-of-vocabulary* (OOV) words.

**Question**: How does the baseline model handle OOV words?

**Exercise**: Improve support for the OOV words in the baseline model.

## Contextualisation

Contextualisation is a technique of transforming input embeddings (in our case,
word embeddings) to contextualized embeddings.  Formally, a contextualisation
module takes on input a sequence of embedding vectors, and outputs a sequence
of contextualised embedding vectors.  The output sequence has *the same length*
as the input sequence, but the *size of word embeddings can change*.

### LSTM

One of the contextualisation techniques is based on *recurrent neural networks*
(RNNs), *long short-term memory* ([LSTM][lstm]) RNNs in particular.  Here's an
`nn.Module` which encapsulates an LSTM transforming embeddings of a given input
size to contextualised embeddings a given output size.
```python
class SimpleLSTM(nn.Module):

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
	self.lstm = nn.LSTM(input_size=inp_size, hidden_size=out_size)

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
```

Note that the `forward` method of an [nn.LSTM][nn-lstm] module takes a
3-dimensional tensor on input.  Its second dimension is the size of a
*mini-batch*, and it allows to process with an LSTM several input sentences in
parallel.  In practice, input sentences have different lengths, and are easier
to be processed when stored in a [packed sequence][packed-seq].

**Exercise**: Extend the baseline model with the `SimpleLSTM` module and see if
you can obtain better performance on the dev set.

**Exercise**: Imlement a bidirectional variant of `SimpleLSTM` (you can call it
`SimpleBiLSTM`).  Have a look at the list of the [nn.LSTM][nn-lstm] module's
available hyper-parameters.  See if it improves the performance of the model.

<!--
### Convolution

### Recurent networks

### Transformer
-->


## Footnotes

<a name="footnote1">1</a>: The sample consists of the first 1000 sentences from
[UD_German-HDT-master/de_hdt-ud-train-a-1.conllu](https://github.com/UniversalDependencies/UD_German-HDT/blob/23f2f1d5ce1621611604c39c9e1069448ec2eb39/de_hdt-ud-train-a-1.conllu).



[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[dataset]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi20/data/UD_German-HDT-sample.zip "UD_German-HDT sample dataset"
[UD_German-HDT-repo]: https://github.com/UniversalDependencies/UD_German-HDT
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
