# Code optimization

This document describes a couple of code optimization techniques (batching,
vectorization, GPU support) which allow to speed up PyTorch applications and
the training process in particular.

The code from the [previous session](../pretrained), relying on a pre-trained
BERT embedding model, is used as a starting point for this document.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Batching](#batching)
  - [Adapting the model](#adapting-the-model)
    - [Biaffine](#biaffine)
    - [BiLSTM](#bilstm)
    - [Joint model](#joint-model)
  - [Adapting the training process](#adapting-the-training-process)
    - [Accuracy](#accuracy)
    - [Batch loss](#batch-loss)
    - [Batch loader and training](#batch-loader-and-training)
- [GPU support](#gpu-support)

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

#### BiLSTM

```python
class BiLSTM(nn.Module):
    '''Contextualise the input sequence of embedding vectors using
    bidirectional LSTM.

    Type: PackedSequence -> PackedSequence

    Example
    -------

    Sample input (randomized):
    >>> ns = torch.tensor([4, 5, 2])        # Sentence lengths
    >>> Din = 10                            # Input embedding size
    >>> Dout = 20                           # Output embedding size
    >>> lstm = BiLSTM(Din, Dout)            # BiLSTM module
    >>> xs = [                              # Sample sentences
    ...     torch.randn(n, Din)
    ...     for n in ns
    ... ]
    >>> seq = rnn.pack_sequence(xs,         # Create packed sequence
    ...             enforce_sorted=False)

    Make sure that forward1 and forward give the same results
    for all inputs in the batch:
    >>> out_seq = lstm.forward(seq)
    >>> ys, _ns = rnn.pad_packed_sequence(out_seq, batch_first=True)
    >>> assert (ns == _ns).all()
    >>> for i in range(len(xs)):
    ...    y1 = lstm.forward1(xs[i])
    ...    y2 = ys[i][:ns[i]]
    ...    assert torch.isclose(y1, y2, atol=1e-6).all()
    '''

    def __init__(self, inp_size: int, out_size: int, **kwargs):
        super().__init__()
        assert out_size % 2 == 0, "Output size have to be even"
        self.lstm = nn.LSTM(
            input_size=inp_size,
            hidden_size=out_size // 2,
            bidirectional=True,
            **kwargs    # Other keyword arguments, if any
        )

    def forward1(self, xs: Tensor) -> Tensor:
        out, _ = self.lstm(xs.view(xs.shape[0], 1, xs.shape[1]))
        return out.view(out.shape[0], out.shape[2])

    def forward(self, seq: rnn.PackedSequence) -> rnn.PackedSequence:
        return self.lstm(seq)[0]
```

#### Joint model

Concerning the `Joint` PyTorch module, the following steps are required to
adapt it to batching:
* Replace `SimpleBiLSTM` with `BiLSTM`
* Implement batching-enabled version of `forward`:
```python
    def forward(self, xs: List[EncInp]) -> Tuple[Tensor, Tensor]:
        # Create a tensor with sentence lengths
        ns = torch.tensor([len(x) for x in xs])
        # Convert the input list to a packed sequence
        seq = rnn.pack_sequence(xs, enforce_sorted=False)
        # Contextualize all sentences in the sequence
        ctx = self.context(seq)
        # Convert the sequence to a packed representation
        embs, _ns = rnn.pad_packed_sequence(ctx, batch_first=True)
        # Check that the sentence length match, just in case
        assert (ns == _ns).all()
        # Calculate and return the scores
        pos_scores = self.score_pos(embs)
        dep_scores = self.score_dep(embs)
        return (pos_scores, dep_scores)
```
* The single-sentence variant `forward1` can be implemented in terms of
  `forward`:
```python
    def forward1(self, xs: EncInp) -> Tuple[Tensor, Tensor]:
        pos_scores, dep_scores = self.forward([xs])
        return (pos_scores[0], dep_scores[0])
```
* The `tag` and `parse` methods can be updated to use `foward1` instead of
  `forward`
* Finally, the documentation string should be updated to account for the
  changes

At this point, it should be possible to test the model on sample data to make
sure there are no runtime exceptions.


### Adapting the training process

#### Accuracy

Concerning the training process, the accuracy functions can be updated by
simply explicitly calling `forward1`, which means they could be further
optimized but let's leave that aside -- our goal is to speed up training, not
necessarily accuracy calculation.

#### Batch loss

The loss of the model should be now calculated on a batch, i.e., on a list of
input/output pairs.  We thus need a function which does precisely that:
```python
def batch_loss_base(model: Joint, batch: List[Tuple[EncInp, EncOut]]):
    '''Cumulative cross-entropy loss of the model on the given dataset.

    Parameters
    ----------
    model : Joint
        Joint POS tagging / dependency parsing model
    batch : List[Tuple[EncInp, EncOut]]
        List of int-encoded input/output pairs
    '''
    # Use cross-entropy loss as training criterion
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # Create lists of input sentences and target values, respectively
    xs = [x for x, _y in batch]
    ys = [y for _x, y in batch]
    # Sentence length in the batch
    ns = list(map(len, xs))
    # Apply the model to the input
    pos_scores_batch, dep_scores_batch = model(xs)
    # Calculate the loss values sentence by sentence
    batch_loss = 0.0
    for i in range(len(xs)):
        # Length of the i-th sentence
        n = ns[i]
        # POS scores and dependency scores for the i-th sentence
        pos_scores = pos_scores_batch[i][:n]
        dep_scores = dep_scores_batch[i][:n][:n+1]
        # Targets POS tags and dependencies
        pos_gold, dep_gold = ys[i]
        # Apply the criterion and update the cumulative batch loss
        batch_loss += criterion(pos_scores, pos_gold) + \
            criterion(dep_scores, dep_gold)
    return batch_loss
```
That's a rather naive implementation, which can be replaced by one that is
truly vectorized:
```python
def batch_loss(model: Joint, batch: List[Tuple[EncInp, EncOut]]):
    '''Cumulative cross-entropy loss of the model on the given dataset.

    Parameters
    ----------
    model : Joint
        Joint POS tagging / dependency parsing model
    batch : List[Tuple[EncInp, EncOut]]
        List of int-encoded input/output pairs
    '''
    # Use cross-entropy loss as training criterion
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    # Create lists of input sentences and target values, respectively
    xs = [x for x, _y in batch]
    ys = [y for _x, y in batch]
    # Gold POS tags and dependencies, padded
    pos_gold = pad([pos for pos, _dep in ys], padding_value=-1)
    dep_gold = pad([dep for _pos, dep in ys], padding_value=-1)
    # Apply the model to the input
    pos_scores, dep_scores = model(xs)
    # Calculate the POS-related cumulative loss
    pos_loss = criterion(pos_scores.permute(0, 2, 1), pos_gold)
    dep_loss = criterion(dep_scores.permute(0, 2, 1), dep_gold)
    return pos_loss + dep_loss
```
The vectorized version relies on a custom padding function:
```python
def pad(xs: List[Tensor], padding_value) -> Tensor:
    '''Pad and stack a list of tensors.

    Parameters
    ----------
    xs : List[Tensor]
        List of tensors with the same number of dimensions and the same dtype.
        Their sizes along the first dimension can differ, but the remaining
        dimensions must be the same.
    padding_value
        Value used for padding (of the same dtype as the tensors in `xs`)

    Examples
    --------
    >>> x = torch.tensor([0, 1, 2])
    >>> y = torch.tensor([3, 4, 5, 6])
    >>> z = torch.tensor([7])
    >>> pad([x, y, z], padding_value=-1)
    tensor([[ 0,  1,  2, -1],
            [ 3,  4,  5,  6],
            [ 7, -1, -1, -1]])
    >>> xs = torch.tensor([[0, 1], [3, 4]])
    >>> ys = torch.tensor([[5, 6], [7, 8], [9, 10]])
    >>> pad([xs, ys], padding_value=-1)[0]
    tensor([[ 0,  1],
            [ 3,  4],
            [-1, -1]])
    >>> pad([xs, ys], padding_value=-1)[1]
    tensor([[ 5,  6],
            [ 7,  8],
            [ 9, 10]])

    The size of the input tensors can differ only along the first dimension:
    >>> zs = torch.tensor([[5, 6, 7]])
    >>> pad([xs, zs], padding_value=-1) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    RuntimeError: ...
    '''
    seq = rnn.pack_sequence(xs, enforce_sorted=False)
    ys, _ns = rnn.pad_packed_sequence(seq,
        batch_first=True, padding_value=padding_value)
    return ys
```
The vectorized version and the naive one provide the exactly same results,
which can be checked by for instance implementing a dedicated test based on
randomized input/output data and random model (which returns random scores; the
actual Joint model could be used, too).

#### Batch loader and training

Finally, we need to update the training procedure.  To this end, we will use a
*batch loader*, which allows to generate a stream of batches (non-overlapping
subsets) from a gives dataset.
```python
def batch_loader(data_set, batch_size: bool, shuffle=False) -> DataLoader:
    """Create a batch data loader from the given data set.

    Using PyTorch Datasets and DataLoaders is especially useful when working
    with large datasets, which cannot be stored in the computer memory (RAM)
    all at once.

    Let's create a small dataset of numbers:
    >>> data_set = range(5)
    >>> for elem in data_set:
    ...     print(elem)
    0
    1
    2
    3
    4

    The DataLoader returned by the batch_loader function allows to
    process the dataset in batches.  For example, in batches of
    2 elements:
    >>> bl = batch_loader(data_set, batch_size=2, shuffle=False)
    >>> for batch in bl:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4]

    The last batch is of size 1 because the dataset has 5 elements in total.
    You can iterate over the dataset in batches over again:
    >>> for batch in bl:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4]

    For the sake of training of a PyTorch model, it may be better to shuffle
    the elements each time the stream of batches is created.
    To this end, use the `shuffle=True` option.
    >>> bl = batch_loader(data_set, batch_size=2, shuffle=True)

    DataLoader "visits" each element of the dataset once.
    >>> sum(len(batch) for batch in bl) == len(data_set)
    True
    >>> set(x for batch in bl for x in batch) == set(data_set)
    True
    """
    return DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=shuffle
    )
```
The training procedure itself can be then updated by:
* Creating the batch loader on top of the training set
```python
    ...
    # Create data loader for the training set
    train_dl = batch_loader(train_data, batch_size=batch_size, shuffle=True)
    ...
```
where `batch_size` should be a parameter of the `train` function.
* Instead of looping over the individual elements of the training set, we
  should loop over the subsequent batches provided by the batch loader:
```python
        ...
        for batch in train_dl:
            # Calculate the loss
            batch_loss = loss(model, batch)
            # Update the total loss on the training set (used
            # for reporting)
            total_loss += batch_loss.item()
            # Calculate the gradients using backpropagation
            batch_loss.backward()
            # Update the parameters along the gradients
            optim.step()
            # Zero-out the gradients
            optim.zero_grad()
        ...
```

## GPU support

**TODO**



[fasttext-models]: https://fasttext.cc/docs/en/crawl-vectors.html#models "Official fastText models for 157 languages"
[fasttext-en-100]: https://user.phil.hhu.de/~waszczuk/treegrasp/fasttext/cc.en.100.bin.gz
[fasttext-python-usage-overview]: https://fasttext.cc/docs/en/python-module.html#usage-overview
[fasttext-reduce-dim]: https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
[bert-as-service]: https://github.com/hanxiao/bert-as-service
[bert-small-models]: https://github.com/google-research/bert/#bert

