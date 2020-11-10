# Encoding and embedding

The data to which we can apply deep learning can come in variety of different
formats.  It can be a list of non-tokenized text chunks coupled with their
sentiments:
```
I love it => positive
I hate it => negative
I don't hate it => neutral
I hate it...  Just kidding!  I actually like it. => positive
```
A list of translation pairs:
```
Only the Thought Police mattered. ||| Zu fürchten war nur die Gedankenpolizei.
That was very true, he thought. |||  Das war sehr richtig, dachte er.
He loved Big Brother. ||| Er liebte den Großen Bruder.
```
Or a list of word-segmented, morphologically-tagged, and dependency-parsed
sentences in a dedicated format such as [CoNLL-U][conllu]:
```
# text = Only the Thought Police mattered.
1	Only	only	ADV	RB	_	4	advmod	_	_
2	the	the	DET	DT	_	4	det	_	_
3	Thought	thought	PROPN	NNP	_	4	compound	_	_
4	Police	police	PROPN	NNP	_	5	nsubj	_	_
5	mattered	matter	VERB	VBD	_	0	root	_	_
6	.	.	PUNCT	.	_	5	punct	_	_

# That was very true, he thought.
1	That	that	PRON	DT	_	4	nsubj	_	_
2	was	be	AUX	VBD	_	4	cop	_	_
3	very	very	ADV	RB	_	4	advmod	_	_
4	true	true	ADJ	JJ	_	0	root	_	_
5	,	,	PUNCT	,	_	4	punct	_	_
6	he	he	PRON	PRP	_	7	nsubj	_	_
7	thought	think	VERB	VBD	_	4	parataxis	_	_
8	.	.	PUNCT	.	_	4	punct	_	_

# text = He loved Big Brother.
1	He	he	PRON	PRP	_	2	nsubj	_	_
2	loved	love	VERB	VBD	_	0	root	_	_
3	Big	big	PROPN	NNP	_	4	compound	_	_
4	Brother	brother	PROPN	NNP	_	2	obj	_	_
5	.	.	PUNCT	.	_	2	punct	_	_

```
In any case, the data has the structure of a list of (input, output) pairs,
which need to be encoded as tensors for subsequent PyTorch processing.

**Note**: Sometimes the dataset is already stored in numerical form, in which
case encoding might not be necessary.  However, this is rarely the case in NLP,
where input/output typically has a structured, categorical form.

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
# text = Only the Thought Police mattered.
1	Only	only	ADV	RB	_	4	advmod	_	_
2	the	the	DET	DT	_	4	det	_	_
3	Thought	thought	PROPN	NNP	_	4	compound	_	_
4	Police	police	PROPN	NNP	_	5	nsubj	_	_
5	mattered	matter	VERB	VBD	_	0	root	_	_
6	.	.	PUNCT	.	_	5	punct	_	_

# That was very true, he thought.
1	That	that	PRON	DT	_	4	nsubj	_	_
2	was	be	AUX	VBD	_	4	cop	_	_
3	very	very	ADV	RB	_	4	advmod	_	_
4	true	true	ADJ	JJ	_	0	root	_	_
5	,	,	PUNCT	,	_	4	punct	_	_
6	he	he	PRON	PRP	_	7	nsubj	_	_
7	thought	think	VERB	VBD	_	4	parataxis	_	_
8	.	.	PUNCT	.	_	4	punct	_	_

# text = He loved Big Brother.
1	He	he	PRON	PRP	_	2	nsubj	_	_
2	loved	love	VERB	VBD	_	0	root	_	_
3	Big	big	PROPN	NNP	_	4	compound	_	_
4	Brother	brother	PROPN	NNP	_	2	obj	_	_
5	.	.	PUNCT	.	_	2	punct	_	_

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
# => TokenList<Only, the, Thought, Police, mattered, .>
# => TokenList<That, was, very, true, ,, he, thought, .>
# => TokenList<He, loved, Big, Brother, .>

# Create the extracted dataset
data: List[Tuple[Inp, Out]] = []
for token_list in conllu.parse(raw_data):
    data.append(extract(token_list))

for inp, out in data:
    print(list(zip(inp, out)))
# => [('Only', 'ADV'), ('the', 'DET'), ('Thought', 'PROPN'), ('Police', 'PROPN'), ('mattered', 'VERB'), ('.', 'PUNCT')]
# => [('That', 'PRON'), ('was', 'AUX'), ('very', 'ADV'), ('true', 'ADJ'), (',', 'PUNCT'), ('he', 'PRON'), ('thought', 'VERB'), ('.', 'PUNCT')]
# => [('He', 'PRON'), ('loved', 'VERB'), ('Big', 'PROPN'), ('Brother', 'PROPN'), ('.', 'PUNCT')]
```

**Note**: The type annotations in the code above (such as `Inp`, `Out`,
`List[Tuple[Inp, Out]]`) are optional, you may skip them in your code.
However, they help to document the individual functions and in general make the
code cleaner.  Additionally, if you use the `mypy` linter, type annotations may
help you detecting problems with the code.

## Preprocessing

This is a good moment to implement additional pre-processing.  We could for
instance lower-case all input words to make sure that vector representations of
words are case-insensitive (this is just an example, lower-casing is not really
necessary with modern pre-trained embedding methods).
```python
def preprocess(inp: Inp) -> Inp:
    """Lower-case all words in the input sentence."""
    return [x.lower() for x in inp]

# Apply the pre-processing function to the dataset
for i in range(len(data)):
    inp, out = data[i]
    data[i] = preprocess(inp), out

for inp, _ in data:
    print(inp)
# => ['only', 'the', 'thought', 'police', 'mattered', '.']
# => ['that', 'was', 'very', 'true', ',', 'he', 'thought', '.']
# => ['he', 'loved', 'big', 'brother', '.']
```
In general, pre-processing can be much more important, for instance when the
input is not tokenized.

## Encoding

The next step is to take the extracted, preprocessed dataset and encode it as
PyTorch tensors.

PyTorch is very flexible when it comes to choosing the moment of encoding.  For
instance, we can keep the dataset in its original form and perform encoding
on-the-fly during training.  Alternatively, each (input, output) pair can be
encoded as a pair (input tensor, output tensor) beforehand.  Finally, the
entire dataset can be encoded as a pair of tensors.  We will follow the second
strategy here.

#### Encoding class

Here's an encoding class that can be used to encode categorical values as
indices:
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

    def size(self):
        return len(self.class_to_ix)

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
# => (tensor([0, 1, 2, 3, 4, 5]), tensor([0, 1, 2, 2, 3, 4]))
# => (tensor([ 6,  7,  8,  9, 10, 11,  2,  5]), tensor([5, 6, 0, 7, 4, 5, 3, 4]))
# => (tensor([11, 12, 13, 14,  5]), tensor([5, 3, 2, 2, 4]))
```

#### One-hot encoding

A traditional strategy to encode categorical values (in our case: input words
on the one hand, and output POS tags on the other hand) as vectors is to use
one-hot encoding.  This is not the preferred method to use in PyTorch (probably
because it uses lots of memory for no good reason), but you can transform the
indices to one-hot vectors as described in an answer to this [this
question](https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321).


## Embedding

*Word embedding* is the process of transforming words (or word indices) into
word vector representations.  There are various embedding methods, differing
notably in:
* What is embedded: words, sub-words, the underlying characters, POS tags, ...
* Whether pre-training is used (fastText, BERT) or not (custom, task-specific
  embeddings)
* Whether the embeddings are context-sensitive (ELMo, BERT) or not (glove,
  fastText)

An important point is that we can decide to **adapt the word embeddings** in a
task-specific way, i.e. make them part of the neural model, or keep them as
are.

We will now implement custom (not pre-trained), word-level, context-insensitive
embeddings, with the task of POS tagging in mind.

#### Method 1 (using one-hot encoding)

**Note**: When you run the code below, do not be surprised to obtain different
values in the randomly generated tensors.

```python
# Create an embedding matrix E of shape K x D, where K is the total number of
# words and D is the embedding size.  For instance, with `D = 5`:
E = torch.randn(word_enc.size(), 5)
print(E)	# => tensor([[-0.3989, -0.4453,  0.4060, -0.8910, -0.3444],
		# =>         [-1.1423,  0.6235,  2.0277,  0.3237, -0.3593],
		# =>         [-1.2349,  0.6406, -1.0205,  0.3681,  1.1595],
		# =>         [-0.0194, -0.8635,  0.2696, -1.5330,  0.7196],
		# =>         [-2.7364,  0.1925,  1.3431, -0.6853,  0.2730],
		# =>         [ 0.2457,  0.1546, -0.0630, -1.3336, -0.2761],
		# =>         [-1.1402, -0.1228,  1.6038, -0.9402,  0.9602],
		# =>         [ 1.3473,  0.5925, -0.7968,  0.9866, -0.2747],
		# =>         [-1.4630, -0.6059,  1.2989, -0.6277,  0.6335],
		# =>         [-0.9366, -0.6316,  0.6680,  0.4417, -1.2921],
		# =>         [ 1.0281, -1.5646,  2.5410, -0.9653,  1.7440],
		# =>         [ 0.7385,  0.3800,  0.0389, -2.2570,  0.2065],
		# =>         [ 0.4360,  1.1240,  0.9782, -0.5362,  0.7155],
		# =>         [ 0.3804,  2.0651,  0.5114, -0.4095,  0.7149],
		# =>         [-0.2233, -0.2494, -1.9650,  0.3513,  2.4496]])

# Transform the encoded input of the second sentence to one-hot vector
x, y = enc_data[1]
print(x)	# => tensor([ 6,  7,  8,  9, 10, 11,  2,  5])
one_hot = torch.nn.functional.one_hot(x, num_classes=word_enc.size())
print(one_hot)  # => tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		# =>         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# =>         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Cast one-hot vector to float and use matrix/matrix product (or @)
torch.mm(one_hot.float(), E)	# equivalently: one_hot.float() @ E
		# => tensor([[-1.1402, -0.1228,  1.6038, -0.9402,  0.9602],
		# =>         [ 1.3473,  0.5925, -0.7968,  0.9866, -0.2747],
		# =>         [-1.4630, -0.6059,  1.2989, -0.6277,  0.6335],
		# =>         [-0.9366, -0.6316,  0.6680,  0.4417, -1.2921],
		# =>         [ 1.0281, -1.5646,  2.5410, -0.9653,  1.7440],
		# =>         [ 0.7385,  0.3800,  0.0389, -2.2570,  0.2065],
		# =>         [-1.2349,  0.6406, -1.0205,  0.3681,  1.1595],
		# =>         [ 0.2457,  0.1546, -0.0630, -1.3336, -0.2761]])
```

#### Method 2 (using indexing)

```python
# Assuming the same embedding matrix `E` of shape K x 5 as above, we can simply
# use the index vector `x` as index
print(E[x]) 	# => tensor([[-1.1402, -0.1228,  1.6038, -0.9402,  0.9602],
		# =>         [ 1.3473,  0.5925, -0.7968,  0.9866, -0.2747],
		# =>         [-1.4630, -0.6059,  1.2989, -0.6277,  0.6335],
		# =>         [-0.9366, -0.6316,  0.6680,  0.4417, -1.2921],
		# =>         [ 1.0281, -1.5646,  2.5410, -0.9653,  1.7440],
		# =>         [ 0.7385,  0.3800,  0.0389, -2.2570,  0.2065],
		# =>         [-1.2349,  0.6406, -1.0205,  0.3681,  1.1595],
		# =>         [ 0.2457,  0.1546, -0.0630, -1.3336, -0.2761]])
```

#### Method 3 (using nn.Embedding)

PyTorch provides a dedicated
[Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
class to store embedding objects.
```python
# Create a new embedding object of shape K x 5
emb = torch.nn.Embedding(word_enc.size(), 5)

# Or from pre-existing embedding matrix
emb = torch.nn.Embedding.from_pretrained(E)
print(emb(x)) 	# => tensor([[-1.1402, -0.1228,  1.6038, -0.9402,  0.9602],
		# =>         [ 1.3473,  0.5925, -0.7968,  0.9866, -0.2747],
		# =>         [-1.4630, -0.6059,  1.2989, -0.6277,  0.6335],
		# =>         [-0.9366, -0.6316,  0.6680,  0.4417, -1.2921],
		# =>         [ 1.0281, -1.5646,  2.5410, -0.9653,  1.7440],
		# =>         [ 0.7385,  0.3800,  0.0389, -2.2570,  0.2065],
		# =>         [-1.2349,  0.6406, -1.0205,  0.3681,  1.1595],
		# =>         [ 0.2457,  0.1546, -0.0630, -1.3336, -0.2761]])
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

### When to embed?

In an actual PyTorch application you may decide to perform embedding at
different stages:
1. Make embedding part of the neural model
2. Embed the entire dataset before training the neural model

The two methods have different trade-offs.  
* Method 1. is the only option if you want to train/adapt the embeddings for a
  given NLP task
* Method 2. may be more convenient (and faster) if you use pre-trained word
  embeddings

Both methods will also differ in memory requirements.


[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
