# Training by gradient descent

### :construction: Work In Progress :construction:

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [POS tagging dataset](#pos-tagging-dataset)
- [Baseline model](#baseline-model)
- [Loss](#loss)
    - [Footnotes](#footnotes)
- [Backward calculation](#backward-calculation)
- [Gradient descent](#gradient-descent)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## POS tagging dataset

We will use the POS-tagging dataset from the [encoding
session](../encoding/README.md), after extraction, pre-processing, and
encoding.  All this is implemented in the [data.py](data.py) module.


## Baseline model

As a baseline model, let's apply an [nn.Linear][linear] to directly score the
word embedding representations.
```python
import torch
import torch.nn as nn

import data
from data import enc_data

# For reproducibility
torch.manual_seed(0)

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
# First encoded (input, output) pair
enc_data[0]
# => (tensor([0, 1, 2, 3, 4, 5]), tensor([0, 1, 2, 2, 3, 4]))

# Apply the baseline model to the first word of the first sentence
baseline(enc_data[0][0][0])
# => tensor([-0.9836,  0.4314,  0.5208, -0.2047, -0.5286,  0.7035, -0.2217, -1.3357],
# =>        grad_fn=<AddBackward0>)

# Apply the baseline model to the input of the first sentence
baseline(enc_data[0][0])
# => tensor([[-0.9836,  0.4314,  0.5208, -0.2047, -0.5286,  0.7035, -0.2217, -1.3357],
# =>         [-0.9657,  0.7258, -0.7684,  0.9671, -0.1222,  1.2167, -1.2175, -0.8721],
# =>         [-0.9120, -0.8854,  1.1599,  0.0689,  0.2130, -1.0645, -0.7430,  0.1419],
# =>         [-0.9182, -1.4884,  1.1120,  1.0361,  0.4861, -1.3754, -1.4426,  0.4466],
# =>         [ 1.2594, -0.7689,  0.0638, -0.2677,  0.3047,  0.0791, -0.4826, -0.0036],
# =>         [ 0.5270, -0.6693, -0.8103, -0.5349,  1.0109,  0.0976, -0.0234, -0.1674]],
# =>        grad_fn=<AddmmBackward>)
```
You can compare it with the target tensor of class indices:
```python
enc_data[0][1]
# => tensor([0, 1, 2, 2, 3, 4])
```
In this particular case, the model correctly produces the highest scores
(`1.1599`, `1.1120` and `1.0109`) on positions `2`, `2`, and `4` for the 3rd,
4th, and 6th words, respectively.  This is by pure luck (and when you run the
code the numbers will differ due to random initialization), but we can train
the model to provide predictions closer to the gold truth.

TODO: Update the example above.


## Loss

To train a model, we need a measure of how well it matches the data.  This
measure is customarily called a *loss* function, although other names for the
same concept exist (e.g., *objective* function).

One of the most useful loss functions in NLP<sup>[1](#footnote1)</sup> is
*cross-entropy loss*, which measure the [cross entropy][cross-entropy] between:
* the predicted distribution of target classes (the one predicted by the model)
* the target, ,,gold'' distribution of classes (as specified in the dataset)

In PyTorch, cross entropy is implemented with the
[CrossEntropyLoss][cross-entropy-loss] class.  It is a [neural module][module]
which represents a function with two arguments:
1. Float tensor of shape `N x C`, where `C` is the number of target classes and `N` is the number of input words<sup>[2](#footnote2)</sup>
1. Integer tensor of shape `N`

Tensor 1. corresponds to the predicted distribution of target classes.  It
takes the form of a matrix of scores assigned to the individual input words.
Note that this is precisely what the baseline model gives on output.

Tensor 2. represents the target classes (as indices) for the individual input
words.  Note this is precisely what is stored as output in the dataset.
```python
loss = nn.CrossEntropyLoss()
for x, y in enc_data:
    print(loss(baseline(x), y))
# => tensor(2.1675, grad_fn=<NllLossBackward>)
# => tensor(1.8645, grad_fn=<NllLossBackward>)
# => tensor(2.2602, grad_fn=<NllLossBackward>)
```

## Backward calculation

A distinguishing feature of PyTorch is that it automatises the gradient
calculation process.  Once we calculate the value of the loss function that we
want to minimize in the *forward* pass, we can perform the *backward* pass.
This will calculate the gradients of all the parameters of the
model.<sup>[3](#footnote3)</sup>
```python
# The gradients of the parameters of the embedding and the linear scoring
# modules are initially set to None
baseline[0].weight.grad is None
# => True
baseline[1].weight.grad is None
# => True

# Let's calculate the gradient of the loss for the first input, output pair
x, y = enc_data[0]
print(loss(baseline(x), y))
# => tensor(2.1675, grad_fn=<NllLossBackward>)
loss(baseline(x), y).backward()

# We can now inspect the gradients
print(baseline[0].weight.grad)
# => tensor([[ 0.0069, -0.0296,  0.0343,  0.0020, -0.0318,  0.0511, -0.0532, -0.0145,
# =>           0.0384, -0.0406],
# =>         [-0.0115,  0.0249, -0.0598,  0.0445,  0.0105,  0.0330, -0.0459, -0.0178,
# =>          -0.0407, -0.0170],
# =>                                   .....................
# =>         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
# =>           0.0000,  0.0000],
# =>         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
# =>           0.0000,  0.0000]])

print(baseline[1].weight.grad)
# => tensor([[ 0.2592,  0.1373,  0.0047,  0.1294, -0.0717, -0.1097, -0.0078,  0.3107,
# =>           0.0664,  0.4092],
# =>         [-0.0047, -0.0980, -0.0483, -0.1750, -0.1435,  0.0370,  0.1953,  0.2672,
# =>          -0.0027,  0.0093],
# =>                                   .....................
# =>         [ 0.0357, -0.0782, -0.0400,  0.0445,  0.0548,  0.0296, -0.0410, -0.0745,
# =>           0.0873,  0.0986],
# =>         [ 0.0269, -0.0869, -0.0390,  0.0671,  0.0719,  0.0325, -0.0566, -0.1080,
# =>           0.0885,  0.0847]])
```

TODO:
* Nudge the parameters
* Show that the loss gets lower



## Gradient descent

TODO


## Footnotes

<a name="footnote1">1</a>: It can be applied within the context of POS tagging,
dependency parsing, sentiment analysis, etc.  Possibly also within the context
of neural machin translation.

<a name="footnote2">2</a>: The number of input words `N` can correspond to a
single input sentence, or to the entire dataset.  This really depends on the
training technique, but what is important that
[CrossEntropyLoss][cross-entropy-loss] can be used in either case.

<a name="footnote3">3</a>: As well as other tensors with `requires_grad` set to
`True` which participate in the forward process.





[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[module]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Module.html?highlight=module#torch.nn.Module "PyTorch neural module"
[cross-entropy]: https://en.wikipedia.org/wiki/Cross_entropy "Cross entropy"
[cross-entropy-loss]: https://pytorch.org/docs/1.6.0/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss "Cross entropy loss criterion"
