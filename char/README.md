# Character-level modeling

This documents summarizes the modifications in the [POS tagging code][context]
required to make use of character-level modeling.  Instead of embedding entire
words, we will embed characters, and use a word-level character-based LSTM (or
convolution) to capture word representations. See for instance this paper [this
paper (Fig. 1)][cnn-char-paper] for a graphical representation of the idea of
extracting character-level representations of words.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Encoding characters](#encoding-characters)
  - [Implementation](#implementation)
  - [Example](#example)
    - [Padding](#padding)
- [Embedding characters](#embedding-characters)
- [From characters to words](#from-characters-to-words)
  - [CBOW](#cbow)
  - [LSTM](#lstm)
    - [Optimization](#optimization)
  - [Convolution](#convolution)
- [Footnotes](#footnotes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<!--
## Changelog
-->


## Encoding characters

First let's introduce a new type to represent characters in `data.py`:
```python
# Single character
Char = NewType('Char', str)
```
The next step is to replace the word&#8594;index encoding with
character&#8594;index encoding in `data.py`.  Put differently, we need to
implement encoding-related functions with the following types:
```python
def create_encoders(
    data: List[Tuple[Inp, Out]]
) -> Tuple[Encoder[Char], Encoder[POS]]:
    ...

def encode_with(
    data: List[Tuple[Inp, Out]],
    char_enc: Encoder[Char],
    pos_enc: Encoder[POS]
) -> List[Tuple[List[Tensor], Tensor]]:
    ...
```
Two notable changes are:
* The second element of the result of `create_encoders` is a character-level
  encoder instead of a word-level encoder (marked with `Encoder[Char]`, which
  replaces `Encoder[Word]`)
* The result of `encode_with` is a list of encoded dataset pairs and the first
  element of each pair (which represents an input sentence) is a *list* of
  tensors rather than a tensor

### Implementation

Here are the implementations of the two functions (but **try to implement them
yourself first rather than just copy-paste the code**):
```python
def create_encoders(
    data: List[Tuple[Inp, Out]]
) -> Tuple[Encoder[Char], Encoder[POS]]:
    """Create a pair of encoders, for characters and POS tags respectively."""
    char_enc = Encoder(
        Char(char) for inp, _ in data   # Using `Chat(..)` for the sake
                                        # of type-checking
        for word in inp
        for char in word
    )
    pos_enc = Encoder(pos for _, out in data for pos in out)
    return (char_enc, pos_enc)

def encode_with(
    data: List[Tuple[Inp, Out]],
    char_enc: Encoder[Char],
    pos_enc: Encoder[POS]
) -> List[Tuple[List[Tensor], Tensor]]:
    """Encode a dataset using given input character and output POS tag encoders."""
    enc_data = []
    for inp, out in data:
        enc_inp = [
            torch.tensor([char_enc.encode(Char(c)) for c in word])
            for word in inp
        ]
        enc_out = torch.tensor([pos_enc.encode(pos) for pos in out])
        enc_data.append((enc_inp, enc_out))
    return enc_data
```

### Example

We can now test the encoding implementation:
```python
# PyTorch modules
import torch
import torch.nn as nn

# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
char_enc, pos_enc = create_encoders(train_data)

# Check encoding on sample characters
char_enc.encode('a')    # => 15
char_enc.encode('b')    # => 5
char_enc.encode('c')    # => 12

# Encode the train set
enc_train = encode_with(train_data, char_enc, pos_enc)

# Inspect the first input sentence
for word, enc_word in zip(train_data[0][0], enc_train[0][0]):
    print(word, "=>", enc_word)
# => Distribution => tensor([0, 1, 2, 3, 4, 1, 5, 6, 3, 1, 7, 8])
# => of => tensor([7, 9])
# => this => tensor([ 3, 10,  1,  2])
# => license => tensor([11,  1, 12, 13,  8,  2, 13])
# => does => tensor([14,  7, 13,  2])
# => not => tensor([8, 7, 3])
# => create => tensor([12,  4, 13, 15,  3, 13])
# => an => tensor([15,  8])
# => attorney => tensor([15,  3,  3,  7,  4,  8, 13, 16])
# => - => tensor([17])
# => client => tensor([12, 11,  1, 13,  8,  3])
# => relationship => tensor([ 4, 13, 11, 15,  3,  1,  7,  8,  2, 10,  1, 18])
# => . => tensor([19])
```

#### Padding

It may be tempting to make `encode_with` return a list of tensor pairs
(`List[Tuple[Tensor, Tensor]]`), but that's actually not trivial since each
word has a different length, so we cannot `torch.stack` the words' tensor
representations.  A workaround is to use `pad_sequence`, but then we would need
to carefully handle the padding value in the rest of the architecture.
```python
from torch.nn.utils.rnn import pad_sequence
xs = [enc_word for enc_word in enc_train[0][0]]
pad_sequence(xs, batch_first=True, padding_value=char_enc.size())
# => tensor([[ 0,  1,  2,  3,  4,  1,  5,  6,  3,  1,  7,  8],
# =>         [ 7,  9, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
# =>                                ...
# =>         [ 4, 13, 11, 15,  3,  1,  7,  8,  2, 10,  1, 18],
# =>         [19, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]])
```

## Embedding characters

We can now adapt the implementation of the neural model by taking care of the
following two points:
* An input sentence is represented by a list of index tensors (one per word)
  rather than a single index tensor; we need to adapt the embedding
  functionality to account for that
* Once we embed characters, each word will be represented by a matrix tensor
  rather than a vector tensor; so we will need one additional module to
  transform character vector representations to word representations

Concerning the first point, it is not possible to use the PyTorch
`nn.Embedding` module with lists -- the argument must be a tensor.
```python
# Create the neural embedding module
emb = nn.Embedding(char_enc.size()+1, 10, padding_idx=char_enc.size())

# Try to apply it to the first input sentence
xs  # => [tensor([0, 1, 2, 3, 4, 1, 5, 6, 3, 1, 7, 8]), ..., tensor([19])]
emb(xs)
# => Traceback (most recent call last):
# =>                    ...
# => TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not list
```
To handle this issue in a simple and modular way, we can introduce the
following ,,higher-order'' mapping module:
```python
class Map(nn.Module):
    """Apply a given module to each element in the list."""

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, xs):
        ys = []
        for x in xs:
            ys.append(self.f(x))
        return ys
```
Here's an example of its usage in combination with character-level embeddings:
```python
# Create the embedding-related modules
emb = nn.Embedding(char_enc.size()+1, 10, padding_idx=char_enc.size())
map_emb = Map(emb)

# Embeddings for the word "Distribution"
emb(xs[0])
# => tensor([[ 1.5856, -1.2990,  0.5764, -1.2626, -1.2547],
# =>                            ...
# =>         [-1.0666,  0.0630, -0.8407, -0.4462, -0.9036]],
# =>        grad_fn=<EmbeddingBackward>)

# Embeddings for all words in the first input sentence
ys = map_emb(xs)
type(ys)
# => <class 'list'>
type(ys[0])
# => <class 'torch.Tensor'>

# Embeddings for "Distribution"
ys[0]
# => tensor([[ 1.5856, -1.2990,  0.5764, -1.2626, -1.2547],
# =>                            ...
# =>         [-1.0666,  0.0630, -0.8407, -0.4462, -0.9036]],
# =>        grad_fn=<EmbeddingBackward>)
assert (ys[0] == emb(xs[0])).all()
```

## From characters to words

Each word in the current architecture is represented as a tensor matrix, with
one row vector per character.  To plug this in the remaining of the existing
architecture for POS tagging we need to cast the word representations
(matrices) to word embeddings (vectors).  We already know several techniques
which allow for that: CBOW, LSTM, and convolution.

### CBOW

The simplest choice -- perhaps overly simplistic, but satisfying all our
interface constraints -- is CBOW, in which the character-level vectors are
summed up to generate the word level representations.  Let's call the
corresponding module `Sum` for a change:
```python
class Sum(nn.Module):
    """Perform torch.sum w.r.t the first dimension."""

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.sum(m, dim=0)
```
Since, in contrast with what we were doing before, we want to apply it at the
level of words and not entire sentences, we need to use `Map` to integrate it
with the remaining architecture:
```python
model = nn.Sequential(
    Map(nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size())),
    Map(Sum()),
    SimpleBiLSTM(50, 200),
    nn.Linear(200, pos_enc.size())
)
```
Alternatively (and equivalently):
```python
model = nn.Sequential(
    Map(nn.Sequential(
        nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size()),
        Sum(),
    )),
    SimpleBiLSTM(50, 200),
    nn.Linear(200, pos_enc.size())
)
```
We can now fill in the remaining pieces (accuracy function, training procedure,
etc. -- everything as it was before) and try to train an actual model.

### LSTM

To use our `SimpleLSTM` instead of CBOW, we take the last output vector of a
unidirectional LSTM as the representation of a word:
```python
class Apply(nn.Module):
    """Apply a given pure function or module."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

model = nn.Sequential(
    Map(nn.Sequential(
        nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size()),
        SimpleLSTM(50, 200),
        Apply(lambda xs: xs[-1]),
    )),
    SimpleBiLSTM(200, 200),
    nn.Linear(200, pos_enc.size())
)
```
Training this model should also bring perceptible accuracy improvements (`~90%`
vs. `~88-89%` achieved previously on the dev set), provided that we take enough
time and epochs.

**TODO**: Make sure about the accuracy improvements!

#### Optimization

At this point the code will have become quite sluggish.  One of the reasons is
that we apply an LSTM to each word in a sentence separately (that's what
`Map(SimpleLSTM(...))` basically means) and it doesn't parallelize well in this
setting, as mentioned before.

To benefit from the LSTM's parallelization capabilities, we have to apply it to
all the words in a sentence in parallel<sup>[1](#footnote1)</sup>.  This can be
achieved by using `PackedSequence`s, for example like this:
```python
import torch.nn.utils.rnn as rnn

class MapLSTM(nn.Module):
    """Variant of SimpleLSTM which works with packed sequence representations.

    MapLSTM(...) is roughly equivalent to Map(SimpleLSTM(...)).

    Type: List[Tensor[N x Din]] -> List[Tensor[N x Dout]], where
    * `N` is is the length of an input sequence
    * `Din` is the input embedding size
    * `Dout` is the output embedding size
    """

    def __init__(self, inp_size: int, out_size: int, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=inp_size, hidden_size=out_size,
            bidirectional=False, **kwargs,
        )

    def forward(self, xs):
        '''Apply the LSTM to the input sentence.'''
        seq_inp = rnn.pack_sequence(xs, enforce_sorted=False)
        seq_out, _ = self.lstm(seq_inp)
        seq_unpacked, lens_unpacked = rnn.pad_packed_sequence(seq_out, batch_first=True)
        ys = []
        for sent, n in zip(seq_unpacked, lens_unpacked):
            ys.append(sent[:n])
        return ys
```
Then:
```python
model = nn.Sequential(
    Map(nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size())),
    MapLSTM(50, 200),
    Map(Apply(lambda xs: xs[-1])),
    SimpleBiLSTM(200, 200),
    nn.Linear(200, pos_enc.size())
)
```

### Convolution

We can also use 1d convolution instead of LSTM to obtain word representations.
Let's use *maxpooling* (as described [here][maxpool] and used in [this
paper][cnn-char-paper], see Fig. 1) instead of CBOW on top of the feature
vectors extracted by the convolution module.  Maxpooling means here that we
take the maximum value along each dimension of the feature space (just as CBOW
means we take the sum -- or average -- along each dimension of the feature
space).
```python
class MaxPool(nn.Module):
    """Perform torch.max w.r.t the first dimension."""

    def forward(self, xs):
        return torch.max(xs, dim=0).values

model = nn.Sequential(
    Map(nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size())),
    Map(SimpleConv(50, 200, kernel_size=5)),
    Map(MaxPool()),
    SimpleBiLSTM(200, 200),
    nn.Linear(200, pos_enc.size())
)
```

<!--
#### Optimization

Just as with LSTM, it is faster (although theoretically equivalent) to process
all the words in a sentence in parallel with the convolution module.
-->

## Footnotes

<a name="footnote1">1</a>: Better still, we could apply an LSTM to several
sentences in parallel.  We will look at the technique (called *batching*) which
enables this possibility (but also requires significant changes in the code)
later on.


[context]: https://github.com/kawu/hhu-dl-materials-2020/tree/main/context#contextualisation "Contextualization"
[maxpool]: https://cezannec.github.io/CNN_Text_Classification/#maxpooling-over-time "Maxpooling"
[cnn-char-paper]: https://www.aclweb.org/anthology/P16-1101.pdf "Paper using the convolution for extracting character-level representations of words"
[pad-sequence]: https://pytorch.org/docs/1.6.0/generated/torch.nn.utils.rnn.pad_sequence.html?highlight=pad#torch.nn.utils.rnn.pad_sequence "Padding function"
<!--
[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[UD_English-ParTUT]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi20/data/UD_English-ParTUT.zip "UD_English-ParTUT sample dataset"
[UD_German-HDT-repo]: https://github.com/UniversalDependencies/UD_German-HDT
[RNN]: https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Recurrent_neural_network_unfold.svg "RNN"
[LSTM]: https://colah.github.io/posts/2015-08-Understanding-LSTMs "LSTM"
[nn-lstm]: https://pytorch.org/docs/1.6.0/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM "LSTM nn.Module"
[packed-seq]: https://pytorch.org/docs/1.6.0/generated/torch.nn.utils.rnn.PackedSequence.html?highlight=packedsequence#torch.nn.utils.rnn.PackedSequence "Packed sequence"
-->
