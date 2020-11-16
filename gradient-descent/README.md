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

To train a model, we need a measure of its quality w.r.t. to the data.  This
measure is customarily called a *loss* function, although other names for the
same idea exist (e.g., *objective* function).

One o the most useful loss functions in NLP<sup>1</sup> is *cross-entropy
loss*, which measure the [cross entropy][cross-entropy] between:
* the predicted distribution of target classes (the one predicted by the model
  given the current parameters)
* the target distribution of classes (as specified in the dataset)

<sup>1</sup>It applies within the context of POS tagging, dependency parsing,
sentiment analysis, etc.  Possibly also within the context of neural machin
translation.



## Gradient descent



[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[cross-entropy]: https://en.wikipedia.org/wiki/Cross_entropy "Cross entropy"
