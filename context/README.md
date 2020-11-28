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

1. Download and unzip a 1000-sentences sample UD dataset (a fragment of
   [`UD_German-HDT`][UD_German-HDT-repo]<sup>[1](#footnote1)</sup>) from
   [here][dataset]
1. The [data.py](data.py) module allows to read and encode the downloaded
   dataset

```python
TODO
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

TODO:
* Explain and implement accuracy
* Show the result of the baseline model on the dev set

POS tagging accuracy is defined as the percentage of words for which the
classifier predicts the correct POS tag.  One way to implement it is:

## Contextualisation

### Convolution

### Recurent networks

### Transformer


## Footnotes

<a name="footnote1">1</a>: The sample consists of the first 1000 sentences from
[UD_German-HDT-master/de_hdt-ud-train-a-1.conllu](https://github.com/UniversalDependencies/UD_German-HDT/blob/23f2f1d5ce1621611604c39c9e1069448ec2eb39/de_hdt-ud-train-a-1.conllu).



[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[dataset]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi20/data/UD_German-HDT-sample.zip "UD_German-HDT sample dataset"
[UD_German-HDT-repo]: https://github.com/UniversalDependencies/UD_German-HDT

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
