# Contextualization

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Dataset

TODO:
* Increate the size of the dataset (use a fragment of ParTUT?)
* Split it to train/dev/test
* Read data from disk?


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

TODO:
* Explain and implement accuracy
* Show the result of the baseline model on the dev set

## Contextualisation methods

### Convolution

### Recurent networks

### Transformer


## Footnotes




[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"

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
