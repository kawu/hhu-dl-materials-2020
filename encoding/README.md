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

## Extraction and preporcessing

Before we start encoding the dataset into the tensor form, it is usually
convenient to extract the relevant data and (if necessary) to perform
additional pre-processing.  Let's say we only want to perform
[UPOS](https://universaldependencies.org/u/pos/index.html) tagging, using the
pre-segmented words on input.
```python
from typing import Tuple, List

import conllu

data = """
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
You can then use the `conllu.parse` and the `extract` functions to extract the
relevant information.
``python
for sent in conllu.parse(data):
    print(sent)
    print(extract(sent))
# => TokenList<The, quick, brown, fox, jumps, over, the, lazy, dog, .>
# => (['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'])
```

## Encoding


[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
