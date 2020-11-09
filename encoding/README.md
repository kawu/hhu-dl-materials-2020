# Encoding and embedding

TODO

## Data

The data to which we can apply deep learning can come in variety of different
formats.  It can be a list of non-tokenized text chunks coupled with their
sentiments:
```
I love it => positive
I hate it => negative
I don't hate it => neutral
I hate it...  Just kidding!  I actually like it. => positive
```
Or a list of translation pairs:
```
1st German sentence ||| 1st English sentence
2st German sentence ||| 2st English sentence
```
Or maybe a list of word-segmented, morphologically-tagged, and
dependency-parsed sentences in a dedicated format such as [CoNLL-U][conllu]:
```
# text = The quick brown fox jumps over the lazy dog.
1   The     the    DET    DT   Definite=Def|PronType=Art   4   det     _   _
2   quick   quick  ADJ    JJ   Degree=Pos                  4   amod    _   _
3   brown   brown  ADJ    JJ   Degree=Pos                  4   amod    _   _
4   fox     fox    NOUN   NN   Number=Sing                 5   nsubj   _   _
5   jumps   jump   VERB   VBZ  Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   0   root    _   _
6   over    over   ADP    IN   _                           9   case    _   _
7   the     the    DET    DT   Definite=Def|PronType=Art   9   det     _   _
8   lazy    lazy   ADJ    JJ   Degree=Pos                  9   amod    _   _
9   dog     dog    NOUN   NN   Number=Sing                 5   nmod    _   SpaceAfter=No
10  .       .      PUNCT  .    _                           5   punct   _   _

```
In any case, the data has the structure of a list of (input, output) pairs,
which need to be encoded as tensors for subsequent PyTorch processing.

#### CoNLL-U

Let's focus on the CoNLL-U example.  There is a [conllu
library](https://pypi.org/project/conllu/) which can be used to parse it in
Python.  To install it, enter in the VSCode terminal (with the `dlnlp`
environment activated):
```
pip install conllu
```

## Extraction

Before we start encoding the dataset into the tensor form, it is usually
convenient to extract the relevant data and (if necessary) to perform
additional pre-processing.  Let's say we only want to perform
[UPOS](https://universaldependencies.org/u/pos/index.html) tagging, using the
pre-segmented words on input.
```python
from typing import Tuple, List

import torch
import conllu

raw_data = """
# text = The quick brown fox jumps over the lazy dog.
1   The     the    DET    DT   Definite=Def|PronType=Art   4   det     _   _
2   quick   quick  ADJ    JJ   Degree=Pos                  4   amod    _   _
3   brown   brown  ADJ    JJ   Degree=Pos                  4   amod    _   _
4   fox     fox    NOUN   NN   Number=Sing                 5   nsubj   _   _
5   jumps   jump   VERB   VBZ  Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   0   root    _   _
6   over    over   ADP    IN   _                           9   case    _   _
7   the     the    DET    DT   Definite=Def|PronType=Art   9   det     _   _
8   lazy    lazy   ADJ    JJ   Degree=Pos                  9   amod    _   _
9   dog     dog    NOUN   NN   Number=Sing                 5   nmod    _   SpaceAfter=No
10  .       .      PUNCT  .    _                           5   punct   _   _

"""

# Input type annotation: a list of words
Inp = List[str]

# Output type annotation: a list of POS tags
Out = List[str]

def extract(token_list: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in token_list:
        inp.append(tok["form"])
        out.append(tok["upos"])
    return inp, out
```
You can then use the `conllu.parse` (or `conllu.parse_incr`) and the `extract`
functions to extract the relevant information.
```python
for token_list in conllu.parse(raw_data):
    print(token_list)
# => TokenList<The, quick, brown, fox, jumps, over, the, lazy, dog, .>

# Create the extracted dataset
data: List[Tuple[Inp, Out]] = []
for token_list in conllu.parse(raw_data):
    data.append(extract(token_list))

for inp, out in data:
    print(inp, out)
# => (['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'])
```


**Note**: the type annotations in the code above (such as `Inp` and `Out`) are
optional, you may skip them in your code.  However, they help to document the
behavior of the individual functions and in general make the code cleaner.
Additionally, if you use the `mypy` linter, it will help detecting problems
with your code when it does not respect the types.

## Preprocessing

This is a good moment to implement additional pre-processing.  We could for
instance lower-case all input words to make sure that vector representations of
words are case-insensitive.
```python
def preprocess(inp: Inp) -> Inp:
    """Lower-case all words in the input sentence."""
    return [x.lower() for x in inp]

# Apply the pre-processing function to the dataset
for i in range(len(data)):
    inp, out = data[i]
    data[i] = preprocess(inp), out

for sent in data:
    print(sent)
# => (['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'])
```
In general, pre-processing can be much more important, for instance when the
input is not tokenized.

## Encoding

The next step is to take the extracted dataset:
```python
for sent in data:
    print(sent)
# => (['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'])
```
and encode it as PyTorch tensors.

PyTorch is very flexible when it comes to choosing the moment of encoding.  For
instance, we can keep the dataset in its original form and perform encoding
on-the-fly during training.  Alternatively, each (input, output) pair can be
encoded as a pair (input tensor, output tensor) beforehand.  Finally, the
entire dataset can be encoded as a pair of tensors.  We will follow the second
strategy here.

#### Encoding class

Here's the encoding class I'm currently using when it comes to encoding
categorical values as indices:
```python
class Encoder:

    """Mapping between classes and the corresponding indices.

    >>> classes = ["English", "German", "French"]
    >>> enc = Encoding(classes)
    >>> assert "English" == enc.decode(enc.encode("English"))
    >>> assert "German" == enc.decode(enc.encode("German"))
    >>> assert "French" == enc.decode(enc.encode("French"))
    >>> set(range(3)) == set(enc.encode(cl) for cl in classes)
    True
    >>> for cl in classes:
    ...     ix = enc.encode(cl)
    ...     assert 0 <= ix <= enc.class_num
    ...     assert cl == enc.decode(ix)
    """

    def __init__(self, classes):
        self.class_to_ix = {}
        self.ix_to_class = {}
        for cl in classes:
            if cl not in self.class_to_ix:
                ix = len(self.class_to_ix)
                self.class_to_ix[cl] = ix
                self.ix_to_class[ix] = cl

    def encode(self, cl):
        return self.class_to_ix[cl]

    def decode(self, ix):
        return self.ix_to_class[ix]
```
Using this class, we can encode our dataset as follows:
```python
# Create the encoder fo the input words
word_enc = Encoder(word for inp, _ in data for word in inp)
# Create the encoder for the POS tags
tag_enc = Encoder(pos for _, out in data for pos in out)

# Encode the dataset
enc_data = []
for inp, out in data:
    enc_inp = torch.tensor([word_enc.encode(word) for word in inp])
    enc_out = torch.tensor([tag_enc.encode(pos) for pos in out])
    enc_data.append((enc_inp, enc_out))
    
# We now have our entire dataset encoded with tensors:
for xy in enc_data:
    print(xy)
# => (tensor([0, 1, 2, 3, 4, 5, 0, 6, 7, 8]), tensor([0, 1, 1, 2, 3, 4, 0, 1, 2, 5]))
```

#### One-hot encoding

A traditional strategy to encode categorical values (in our case: input words
on the one hand, and output POS tags on the other hand) as vectors is to use
one-hot encoding.  This is not the preferred method to use in PyTorch (probably
because it uses lots of memory for no very good reason), but you can transform
the indices to one-hot vectors as described in [this
question](https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321).


## Embedding

*Word embedding* is the process of transforming words (or word indices) into
actual word vector representations.  There are many different embedding
methods, differing notably in:
* What is embedded: words, sub-words, the underlying characters
* Whether pre-training is used (fastText) or not (custom, task-specific
  embeddings)
* Whether the embeddings are context-sensitive (ELMo, BERT) or not (glove,
  fastText)

We will now see how to implement word-level, custom (not pre-trained),
context-insensitive embeddings for the task of POS tagging.

#### Method 1 (using one-hot encoding)

**Note**: when you run the code below, do not be surprised to obtain different
values in the randomly generated tensors.

```python
# Create an embedding matrix E of shape N x D, where N is the total number of
# words and D is the embedding size.  For instance, with `D = 5`:
E = torch.randn(word_enc.size(), 5)
E 		# => tensor([[-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [ 0.3565, -0.9042, -0.3476, -0.6734, -0.2240],
		# =>         [ 0.7418,  1.1183, -0.2996, -0.0334, -0.0498],
		# =>         [-0.9738, -0.0465, -1.1639,  0.2371,  1.3033],
		# =>         [-0.3283, -1.4421,  0.8929, -1.3165, -0.3889],
		# =>         [-0.1805,  0.4792, -0.1668, -1.1172, -0.2787],
		# =>         [-1.4763, -0.3729,  0.6232, -1.3880, -0.6965],
		# =>         [-1.4002,  0.5463,  0.6842, -0.2499,  0.2652],
		# =>         [ 0.6933,  0.5644,  0.5488,  0.2362,  0.3567]])

# Transform the encoded input of the first sentence to one-hot vector
x, y = enc_data[0]
x		# => tensor([0, 1, 2, 3, 4, 5, 0, 6, 7, 8])
one_hot = torch.nn.functional.one_hot(x)
one_hot         # => tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0],
		# =>         [0, 1, 0, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 1, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 1, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 1, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 1, 0, 0, 0],
		# =>         [1, 0, 0, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 1, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 1, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Cast one-hot vector to float and use matrix/matrix product (or @)
torch.mm(one_hot.float(), E)	# equivalently: one_hot.float() @ E
		# => tensor([[-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [ 0.3565, -0.9042, -0.3476, -0.6734, -0.2240],
		# =>         [ 0.7418,  1.1183, -0.2996, -0.0334, -0.0498],
		# =>         [-0.9738, -0.0465, -1.1639,  0.2371,  1.3033],
		# =>         [-0.3283, -1.4421,  0.8929, -1.3165, -0.3889],
		# =>         [-0.1805,  0.4792, -0.1668, -1.1172, -0.2787],
		# =>         [-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [-1.4763, -0.3729,  0.6232, -1.3880, -0.6965],
		# =>         [-1.4002,  0.5463,  0.6842, -0.2499,  0.2652],
		# =>         [ 0.6933,  0.5644,  0.5488,  0.2362,  0.3567]])
```

#### Method 2 (using indexing)

```python
# Assuming the same embedding matrix `E` of shape 9 x 5 as above, we can simply
# use the index vector `x` as index
E[x] 		# => tensor([[-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [ 0.3565, -0.9042, -0.3476, -0.6734, -0.2240],
		# =>         [ 0.7418,  1.1183, -0.2996, -0.0334, -0.0498],
		# =>         [-0.9738, -0.0465, -1.1639,  0.2371,  1.3033],
		# =>         [-0.3283, -1.4421,  0.8929, -1.3165, -0.3889],
		# =>         [-0.1805,  0.4792, -0.1668, -1.1172, -0.2787],
		# =>         [-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [-1.4763, -0.3729,  0.6232, -1.3880, -0.6965],
		# =>         [-1.4002,  0.5463,  0.6842, -0.2499,  0.2652],
		# =>         [ 0.6933,  0.5644,  0.5488,  0.2362,  0.3567]])
```

#### Method 3 (using nn.Embedding)

PyTorch provides a dedicated
[Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
class to store embedding objects.
```python
# Create a new embedding object of shape 9 x 5
emb = torch.nn.Embedding(9, 5)

# Or from pre-existing embedding matrix
emb = torch.nn.Embedding.from_pretrained(E)
emb(x) 		# => tensor([[-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [ 0.3565, -0.9042, -0.3476, -0.6734, -0.2240],
		# =>         [ 0.7418,  1.1183, -0.2996, -0.0334, -0.0498],
		# =>         [-0.9738, -0.0465, -1.1639,  0.2371,  1.3033],
		# =>         [-0.3283, -1.4421,  0.8929, -1.3165, -0.3889],
		# =>         [-0.1805,  0.4792, -0.1668, -1.1172, -0.2787],
		# =>         [-0.0358, -0.3950, -1.8251, -0.0101, -0.8462],
		# =>         [-1.4763, -0.3729,  0.6232, -1.3880, -0.6965],
		# =>         [-1.4002,  0.5463,  0.6842, -0.2499,  0.2652],
		# =>         [ 0.6933,  0.5644,  0.5488,  0.2362,  0.3567]])
```
The `Embedding` class provides some additional functionality, for instance
handling out-of-vocabulary words (see the `padding_idx` attribute).  Moreover,
it's an instance of the
[nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module#torch.nn.Module)
class, a base class for all neural network submodules which we will learn more
about later.
<!--
However, in most cases the three methods are interchangeable (although I don't
see any good reasons to use method 1 in practice).
-->



[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
