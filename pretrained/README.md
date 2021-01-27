# Pre-trained embeddings

So far we were using custom (word- or character-level) embeddings, trained as
part of the network in a task-specific way.  An alternative is to use
embeddings pre-trained on large quantities of un-annotated texts in an
unsupervised way.  Using modern pre-trained embeddings is often inevitable when
it comes to achieving state-of-the-art results.  While pre-training is often
very costly, using pre-trained embeddings typically isn't (although memory
requirements of just using an embedding model on a personal computer can be
prohibitive).

Pre-trained embeddings can be classified in two groups: non-contextualized
(e.g. word2vec, GloVe, fastText) and contextualized (e.g. ELMo, BERT).  The
former embed words without context, similarly as the embedding models we were
using so far.   The latter are context-sensitive in that they take the entire
input sentence and produce the embeddings for the individual words which depend
on the context they occur in.  Below we look at two instances of these two
classes -- fastText and BERT -- and show practically how to plug them in a
neural model.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [fastText](#fasttext)
  - [Setup](#setup)
  - [Basic usage](#basic-usage)
  - [Integration](#integration)
    - [Embedding during pre-processing](#embedding-during-pre-processing)
- [BERT](#bert)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## fastText

### Setup

First install the `fastText` Python library by running in the terminal (within
the `dlnlp` environment):
```bash
pip install fasttext
```
The next step is to download a pre-trained model for a particular language,
preferably one of the [models officially distributed on the fastText
website][fasttext-models].  The models are available in two formats: textual
and binary.  In practice the latter is easier to use in a PyTorch application.
In the following we use the official binary model for English, which can be
downloaded on Linux from the command line using:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
```
Note that the file is relatively large (4.2 GB) so the download can take some
time.

**WARNING**: The model is quite memory greedy and I wouldn't recommend using it
if you haven't at least 16 GB of RAM on your personal computer.  An alternative
is to use a `fastText` model with [reduced
dimensionality][fasttext-reduce-dim].  You can download a reduced (with
embeddings of size 100) English model from [here][fasttext-en-100] .  Note that
reducing dimensionality often leads to lower results.


### Basic usage

To test the setup, open the Python interpreter, import the `fasttext` library,
and try embedding some English words.  You can find the overview of the usage
of the Python `fasttext` module [here][fasttext-python-usage-overview].
```python
model = fasttext.load_model("cc.en.300.bin")
model['asparagus']
# => array([ 2.43494716e-02, -1.13698803e-02, -8.15532170e-03,  5.03604002e-02,
# =>                                        ...
# =>        -1.08037302e-02,  9.83848721e-02,  3.61668691e-02,  8.09734687e-03],
# =>       dtype=float32)
```
The output vector is a numpy array of size 300:
```python
model['asparagus'].size     # => 300
type(model['asparagus'])    # => <class 'numpy.ndarray'>
```
You can transform it easily to a PyTorch tensor as follows:
```python
x = torch.tensor(model['asparagus'])
```
Note that the resulting tensor does not ,,require gradient'' by default: 
```python
x.requires_grad             # => False
```
which means that it won't get adapted during training (which is most often the
intended behavior, in contrast with the custom embeddings we were using so
far).

An important advantage of fastText embeddings in contrast with custom
word-level embeddings is that you no longer have to worry about spelling
mistakes or OOV words, thanks to fastText using sub-word information to produce
the resulting vectors (admittedly, using a character-level model to produce
word embeddings also alleviates these issues).
<!--
```python
model['asparags']
# => array([ 1.54658407e-03,  6.71681017e-02,  9.10522602e-03,  6.13789335e-02,
# =>        -7.91550986e-03,  1.44495629e-02,  2.46534199e-02,  3.39101180e-02],
# =>       dtype=float32)
```
-->

### Integration

To integrate the fastText model in our dependency parser / POS tagger, we can
follow one of two strategies:
* Produce the fastText embeddings as part of pre-processing
* Calculate the fastText embeddings as part of the model

In any case, it is no longer necessary to encode input words as integers
(fastText models take strings on input).  The second strategy is the only one
that can be used if you want to adapt the embeddings (i.e. make them parameters
of the neural model).  Below we describe an implementation of the first
strategy.

#### Embedding during pre-processing

Embedding during pre-processing means that we want to apply the fastText model
to replace the words on input:
```python
# Input: a list of words
Inp = List[Word]
```
with the corresponding fastText embeddings (tensors) once in advance.  We can
reuse the `EncInp` type to express that we want to perform embedding during the
stage of encoding (which is performed only once and hence makes part of
pre-processing).
```python
# Encoded input: a matrix of pre-trained embeddings
EncInp = Tensor
```
This will lead to a typing error in the `encode_input` function:
```console
pretrained$ mypy data.py
data.py:99: error: Incompatible return value type (got "List[Tensor]", expected "Tensor")
```
which we can update to use a fastText model instead of a character-level
encoder.
```python
def encode_input(sent: Inp, ft_model) -> EncInp:
    """Embed an input sentence given a fastText model."""
    return torch.tensor([ft_model[word] for word in sent])
```
This will be enough to satisfy the type-checker, since `ft_model` in the code
above is not type-annotated, but to keep the code clean the types and docstring
of the `encode_with` function should be updated as well.

The next step is to modify the main script (`session.py`) in order to account
for the change of the encoding and embedding strategy.  First of all, we can
use underscore to mark the character encoder as unused (this is just a
convention) and load the desired fastText model:
```python
...
import fasttext # type: ignore
...
_char_enc, pos_enc = create_encoders(train_data)

# Load a fastText model
fastText = fasttext.load_model("cc.en.300.bin")   # or "cc.en.100.bin"

# Encode the train and dev sets, using fastText and the POS encoder
enc_train = encode_with(train_data, fastText, pos_enc)
enc_dev = encode_with(dev_data, fastText, pos_enc)
...
```
At this point, we can re-define the `Joint` model, taking into account that:
* We don't need the embedding-related components any more
* Embedding size is 300 (or 100 in case of the reduced model)
* The character-level encoder should be replaced with the fastText model

In particular:
```python
class Joint(nn.Module):
    ...
    def __init__(self,
        ft_model,                   # fastText embedding model
        pos_enc: Encoder[POS],      # Encoder for POS tags
        emb_size: int,              # Embedding size
        hid_size: int               # Hidden size used in LSTMs
    ):
        ...
        self.ft_model = ft_model
        ...
        # Common part of the model: LSTM contextualization
        self.embed = nn.Sequential(
            # NOTE: Embedding components removed (embedding part of pre-processing)
            SimpleBiLSTM(emb_size, hid_size),
        )
```

These are all the changes needed to adapt the model to use pre-trained fastText
embeddings.  We can now proceed to train the model and see the impact on the
resulting accuracy.


## BERT

:construction: work in progress :construction:



[fasttext-models]: https://fasttext.cc/docs/en/crawl-vectors.html#models "Official fastText models for 157 languages"
[fasttext-en-100]: https://user.phil.hhu.de/~waszczuk/treegrasp/fasttext/cc.en.100.bin.gz
[fasttext-python-usage-overview]: https://fasttext.cc/docs/en/python-module.html#usage-overview
[fasttext-reduce-dim]: https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension

