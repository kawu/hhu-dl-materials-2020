# Pre-trained embeddings

:construction: work in progress :construction:

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
- [BERT](#bert)
  - [Setup](#setup-1)
    - [Server](#server)
    - [Client](#client)
  - [Usage](#usage)
    - [Contextualized embeddings](#contextualized-embeddings)
    - [Sentence length](#sentence-length)
    - [Processing tokenized sentence](#processing-tokenized-sentence)
  - [Integration](#integration-1)

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
of the neural model).  Here we describe an implementation of the first
strategy.

<!--
#### Embedding during pre-processing
-->

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
            # NOTE: Embedding components removed
            SimpleBiLSTM(emb_size, hid_size),
        )
```

These are all the changes needed to adapt the model to use pre-trained fastText
embeddings.  We can now proceed to train the model and see the impact on the
resulting accuracy.
```python
train(model, enc_train, enc_dev, loss, pos_accuracy, epoch_num=10, learning_rate=0.001, report_rate=1)
# => @1: loss(train)=4411.780, acc(train)=0.874, acc(dev)=0.850
# => @2: loss(train)=2360.391, acc(train)=0.920, acc(dev)=0.900
# => @3: loss(train)=1814.367, acc(train)=0.938, acc(dev)=0.914
# => @4: loss(train)=1479.676, acc(train)=0.948, acc(dev)=0.920
# => @5: loss(train)=1227.272, acc(train)=0.955, acc(dev)=0.925
# => @6: loss(train)=1023.560, acc(train)=0.961, acc(dev)=0.927
# => @7: loss(train)=865.200, acc(train)=0.964, acc(dev)=0.925
# => @8: loss(train)=736.572, acc(train)=0.969, acc(dev)=0.927
# => @9: loss(train)=604.668, acc(train)=0.971, acc(dev)=0.921
# => @10: loss(train)=526.406, acc(train)=0.974, acc(dev)=0.921
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.0001, report_rate=1)
# => @1: loss(train)=342.685, acc(train)=0.966, acc(dev)=0.769
# => @2: loss(train)=250.953, acc(train)=0.976, acc(dev)=0.774
# => @3: loss(train)=212.318, acc(train)=0.982, acc(dev)=0.775
# => @4: loss(train)=186.850, acc(train)=0.986, acc(dev)=0.771
# => @5: loss(train)=164.101, acc(train)=0.989, acc(dev)=0.770
# => @6: loss(train)=146.069, acc(train)=0.992, acc(dev)=0.775
# => @7: loss(train)=129.453, acc(train)=0.994, acc(dev)=0.771
# => @8: loss(train)=115.474, acc(train)=0.995, acc(dev)=0.767
# => @9: loss(train)=103.844, acc(train)=0.997, acc(dev)=0.768
# => @10: loss(train)=93.190, acc(train)=0.998, acc(dev)=0.772
pos_accuracy(model, enc_dev)
# => 0.922850844966936
```


## BERT

This sections shows how to use [bert-as-service][bert-as-service].

### Setup

#### Server

The server can be installed in a separate conda environment, in order to avoid
dependency issues.
* Create a dedicated conda environment (`bert-as-service` requires tensorflow
  v1, which in turn makes it necessary to downgrade to Python 3.7)
    * `conda create --name bert-service python=3.7`
    * `conda activate bert-service`
* Install tensorflow 1.15 (a dependency of `bert-as-service`)
    * `pip install tensorflow==1.15`
* Install the server, as explained
  [here](https://github.com/hanxiao/bert-as-service#install)
    * `pip install bert-serving-server`

The next step is to download a BERT model.  The
[google-research/bert](https://github.com/google-research/bert/) github
repository provides several pre-trained models.
* Download and extract the ,,tiny'' model (`uncased_L-2_H-128_A-2`); this model
  is very small so it can be used on a standard personal computer without
  problems
* Start the server
    * `bert-serving-start -model_dir uncased_L-2_H-128_A-2 -num_worker=4`

<!--
* Download a tiny BERT model:
    * See `https://huggingface.co/google/bert_uncased_L-4_H-128_A-2#bert-miniatures`
    * And `https://github.com/google-research/bert/`
    * Unpack: `unzip uncased_L-2_H-128_A-2.zip -d uncased_L-2_H-128_A-2`
-->

#### Client

The client should be installed in the main `dlnlp` conda environment with the
following command:
```bash
pip install bert-serving-client
```

### Usage

Once the server is running and the client is installed, you can use it to, for
instance, encode sentences as vectors:
```python
from bert_serving.client import BertClient

# Create the client instance
bc = BertClient()

# Encode three sentences to three vectors
bc.encode(['First do it', 'then do it right', 'then do it better'])
# => array([[-2.19040513e+00,  6.67692840e-01, -1.93513811e+00,
# =>                                 ...
# =>         -1.93510860e-01,  1.02745548e-01],
# =>        [-1.74784052e+00,  7.03274667e-01, -1.61399996e+00,
# =>                                 ...
# =>          4.40993279e-01,  6.39024019e-01],
# =>        [-1.99619067e+00,  8.95147741e-01, -1.73543727e+00,
# =>                                 ...
# =>          2.13959828e-01,  4.03461665e-01]], dtype=float32)
```
In this case, the resulting vectors are numpy arrays of size 128 (due to the
selected model), which can be converted to PyTorch tensors using
`torch.tensor`.
```python
type(bc.encode(['First do it']))        # => <class 'numpy.ndarray'>
bc.encode(['then do it right']).size    # => 128
```

See [the
documentation](https://github.com/hanxiao/bert-as-service#server-and-client-api)
for a full description of the client/server API.


#### Contextualized embeddings

BERT allows to retrieve the contextualized embeddings of the individual words
in a sentence.  To this end, the server has to be started with
`pooling_strategy` set to `NONE` (remember to do that in the `bert-as-service`
environment if you followed the [server setup instructions](#setup-1) above!):
```bash
bert-serving-start -pooling_strategy NONE -model_dir uncased_L-2_H-128_A-2 -num_worker=4
```
The BERT client will then output separate vectors for the individual words:
```python
torch.tensor(bc.encode(['then do it better']).copy()).shape
# => torch.Size([1, 25, 128])
```
The first dimension of the result is the batch-size and the second is the
maximum sequence length (see [below](#sentence-length).  Additionally, the
input sentence is implicitely padded on the left and right with special
symbols:
```python
xs = torch.tensor(bc.encode(['then do it better']).copy()).squeeze(0)
xs[0]   # embedding for '[CLS]'
xs[1]   # embedding for 'then'
xs[2]   # embedding for 'do'
xs[3]   # embedding for 'it'
xs[4]   # embedding for 'better'
xs[5]   # embedding for '[SEP]'
xs[6]   # embedding for padding symbol
```
See [the corresponding
documentation](https://github.com/hanxiao/bert-as-service#getting-elmo-like-contextual-word-embedding)
for more details.

#### Sentence length

By default all sentences are trimmed on the right to a maximum sequence length.
To avoid that, set the `max_seq_len` server option to `NONE`, or to an actual
maximum sentence length in your dataset.

#### Processing tokenized sentence

See https://github.com/hanxiao/bert-as-service#using-your-own-tokenizer.
Beware of tokenization mismatches.


### Integration

Integration of the BERT model (with no pooling strategy, see
[above](#contextualized-embeddings)) is similar to the process of integrating
`fastText`:
* Reuse the `EncInp` type to express that we want to perform embedding during
  pre-processing:
```python
# Encoded input: a matrix of pre-trained embeddings
EncInp = Tensor
```
* Update the `encode_input` function to use BERT:
```python
def encode_input(sent: Inp, bc: BertClient) -> EncInp:
    """Embed an input sentence given a BERT client."""
    # Retrieve the embeddings of the sentence
    xs = torch.tensor(bc.encode([sent], is_tokenized=True).copy()).squeeze(0)
    # Discard [CLS] and [SEP] embeddings
    xs = xs[1:len(sent)+1]
    # Make sure the lenghts match, just in case
    assert len(xs) == len(sent)
    return xs
```
* Update the `encode_with` function (in particular its docstring)
* Update the main script to replace the embedding model with BERT

**Results**: Using the [BERT-Base model][bert-small-models] embeddings in the
joint model allows to reach the POS accuracy of around 90\% and UAS (dependency
accuracy) of around 82.5\% on the dev set.

**TODO**: Check the following:
* Does adding Dropout help?
* Is BiLSTM even necessary in this setting?


[fasttext-models]: https://fasttext.cc/docs/en/crawl-vectors.html#models "Official fastText models for 157 languages"
[fasttext-en-100]: https://user.phil.hhu.de/~waszczuk/treegrasp/fasttext/cc.en.100.bin.gz
[fasttext-python-usage-overview]: https://fasttext.cc/docs/en/python-module.html#usage-overview
[fasttext-reduce-dim]: https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
[bert-as-service]: https://github.com/hanxiao/bert-as-service
[bert-small-models]: https://github.com/google-research/bert/#bert
