# Code optimization

This document describes a couple of code optimization techniques (batching,
vectorization, GPU support) which allow to speed up PyTorch applications and
the training process in particular.

The code from the [previous session](../pretrained), relying on a pre-trained
BERT embedding model, is used as a starting point for this document.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Batching

*Batching* is an optimization which involes processing sentences in batches.
This allows to better benefit from the parallelization capabilities of the
underlying processing unit (CPU, GPU) and thus speed up the training process in
particular.  Batching is related to a more general technique of
[*vectorization*](https://stackoverflow.com/questions/1422149/what-is-vectorization).

To apply this technique to our POS tagger / dependency parser, the following
main steps need to be taken:
* The model, as well as some of its components, need to be
  [adapted](#adapting-the-model) to take on input a batch of inputs rather than
  a single input sentence
* The training procedure needs to be [modified](#adapting-the-training-process)
  so that the loss of the model is calculated on a batch of training
  input/output pairs in each step of [SGD](../gradient-descent)

### Adapting the model

#### Biaffine

The previous version of the `Biaffine` module had type `Tensor[N x D] ->
Tensor[N x (N + 1)]`, where `N` is the input sentence length and `D` is the
embedding size.  The batching-enabled variant of `Biaffine` handles tensors
with one additional dimensions, which allows it to process several sentences in
parallel.  The old forward method for processing single-sentence inputs is
still available under the name `forward1`.  Note that there is a test in the
docstring which makes sure that both versions, `forward1` and `forward`, give
the same results.
```python
class Biaffine(nn.Module):
    '''Calculate pairwise matching scores.

    Type: Tensor[B x N x D] -> Tensor[B x N x (N + 1)]

        where

            * B is the batch size
            * N is the maximum sentence length in the batch

    For a given sequence (matrix) of word embeddings, calculate the matrix of
    pairwise matching scores.

    Example
    -------

    Sample input (randomized):
    >>> ns = torch.tensor([4, 5, 2])        # Sentence lengths
    >>> B = len(ns)                         # Batch size
    >>> D = 10                              # Embedding size
    >>> bia = Biaffine(10)                  # Biaffine module
    >>> xs = torch.randn(B, max(ns), D)     # Sample embeddings

    Make sure that forward1 and forward give the same results
    for all inputs in the batch:
    >>> for i in range(B):
    ...    y1 = bia.forward1(xs[i][:ns[i]])
    ...    y2 = bia.forward(xs)[i][:ns[i],:ns[i]+1]
    ...    assert torch.isclose(y1, y2, atol=1e-6).all()
    '''

    def __init__(self, emb_size: int):
        super().__init__()
        # NOTE: Parameter/components are the same as in the
        # non-batching-enabled variant
        self.depr = nn.Linear(emb_size, emb_size)
        self.hedr = nn.Linear(emb_size, emb_size)
        self.root = nn.Parameter(torch.randn(emb_size))

    def forward1(self, xs: Tensor):
        deps = self.depr(xs)
        heds = torch.cat((self.root.unsqueeze(0), self.hedr(xs)))
        return deps @ heds.t()

    def forward(self, xs: Tensor):
        B = xs.shape[0]    # Batch size
        N = xs.shape[1]    # Maximum sentence length
        D = xs.shape[2]    # Input embedding size

        # Calulate the head/dependent representations
        deps = self.depr(xs)
        root = self.root.view(1, 1, -1).expand(B, 1, -1)
        heds = torch.cat((root, self.hedr(xs)), dim=1)

        # Check the shape (optional)
        assert list(heds.shape) == [B, 1 + N, D]

        # Calculate and return the head scores
        return torch.bmm(deps, heds.permute(0, 2, 1))
```

### Adapting the training process




## GPU support




[fasttext-models]: https://fasttext.cc/docs/en/crawl-vectors.html#models "Official fastText models for 157 languages"
[fasttext-en-100]: https://user.phil.hhu.de/~waszczuk/treegrasp/fasttext/cc.en.100.bin.gz
[fasttext-python-usage-overview]: https://fasttext.cc/docs/en/python-module.html#usage-overview
[fasttext-reduce-dim]: https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
[bert-as-service]: https://github.com/hanxiao/bert-as-service
[bert-small-models]: https://github.com/google-research/bert/#bert

