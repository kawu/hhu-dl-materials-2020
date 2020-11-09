# Encoding and embedding

TODO

## Data

The data to which we can apply deep learning can come in variety of different
formats.  It can be a list of non-tokenized text chunks coupled with their
sentiments:
```
First sentence => neutral
Second sentence => negative
Third sentence => positive
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

def extract(sent: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in sent:
        inp.append(tok["form"])
        out.append(tok["upos"])
    return inp, out
```
You can then use the `conllu.parse` (or `conllu.parse_incr`) and the `extract`
functions to extract the relevant information.
```python
for sent in conllu.parse(raw_data):
    print(sent)
# => TokenList<The, quick, brown, fox, jumps, over, the, lazy, dog, .>

# Create the extracted dataset
data = []
for sent in conllu.parse_incr(raw_data):
    data.append(extract(sent))

for sent in data:
    print(sent)
# => (['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'])
```


**Note**: the type annotations in the code above (such as `Inp` and `Out`) are
optional, you may skip them in your code.  However, they help to document the
behavior of the individual functions and in general make the code clearer.
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
for sent in data:
    sent[0] = preprocess(sent[0])

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
        # Make a set to remove duplicates, just in case
        class_set = set(cl for cl in classes)
        self.class_num = len(class_set)
        self.class_to_ix = {}
        self.ix_to_class = {}
        for (ix, cl) in enumerate(class_set):
            self.class_to_ix[cl] = ix
            self.ix_to_class[ix] = cl

    def encode(self, cl: str) -> int:
        return self.class_to_ix[cl]

    def decode(self, ix: int) -> str:
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
# We now have out entire dataset encoded as tensors:
for xy in enc_data:
    print(xy)
# => TODO
```

#### One-hot encoding

A traditional strategy to encode categorical values (in our case: input words
on the one hand, and output POS tags on the other hand) as vectors is to use
one-hot encoding.  PyTorch does not support it directly, but you can transform
the indices to one-hot vectors as described in [this
question](https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321).



[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
