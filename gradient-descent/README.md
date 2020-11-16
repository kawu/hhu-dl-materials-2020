# Training by gradient descent

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Loss](#loss)
- [Gradient descent](#gradient-descent)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Playground dataset

We will use the POS-tagging dataset from the [encoding
session](../enoding/README.md), after extraction, pre-processing, encoding, and
embedding (TODO: really?).  All this is implemented in the
[data.py](data.py) module.


## Baseline model

As a baseline model, let's apply a [nn.Linear][linear] to directly score the
word embedding representations.
```python
import torch
import torch.nn as nn

import data

baseline = nn.Sequential(
    nn.Embedding(data.word_enc.size(), 10),
    nn.Linear(10, data.pos_enc.size())
)
```
The linear transformation layer *scores* the word vector representations: it
assigns a score for each target class (POS tag, in this case).  The higher the
score is, the more plausible the corresponding POS tag for the corresponding
input word.
```python
TODO: examples
```


## Loss

To train a model, we need a measure of how well it matches the data.  This
measure is customarily called a *loss* function, although other names for the
same concept exist (e.g., *objective* function).

One of the most useful loss functions in NLP<sup>1</sup> is *cross-entropy
loss*, which measure the [cross entropy][cross-entropy] between:
* the predicted distribution of target classes (the one predicted by the model)
* the target, ,,gold'' distribution of classes (as specified in the dataset)

In PyTorch, cross entropy is implemented with the
[CrossEntropyLoss][cross-entropy-loss] class.  It is a [neural module][module]
which represents a function with two arguments:
1. Float tensor of shape `N x C`, where `C` is the number of target classes
1. Integer tensor of shape `N`

* Tensor 1. corresponds to the predicted distribution of target classes.  It
  takes the form of score vectors assigned to the individual input words.  Note
  that this is precisely what the baseline model gives on output.
* Tensor 2. represents the target classes (as indices) for the individual input
  words.  Note this is precisely what is stored as output in the dataset.
```python
TODO: example
```


<sup>1</sup>It can be applied within the context of POS tagging, dependency
parsing, sentiment analysis, etc.  Possibly also within the context of neural
machin translation.


## Backward calculation

TODO


## Gradient descent

TODO



[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[module]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Module.html?highlight=module#torch.nn.Module "PyTorch neural module"
[cross-entropy]: https://en.wikipedia.org/wiki/Cross_entropy "Cross entropy"
[cross-entropy-loss]: https://pytorch.org/docs/1.6.0/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss "Cross entropy loss criterion"
