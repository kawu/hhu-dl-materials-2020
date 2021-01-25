# Pre-trained embeddings

So far we were using custom (word- or character-level) embeddings, trained as
part of the network in a task-specific way.  An alternative is to use
embeddings pre-trained on large quantities of un-annotated texts in an
unsupervised manner.  Using modern pre-trained embeddings is often unavoidable
when it comes to achieving state-of-the-art results.  While their
(pre-)training is often very costly, using them typically is not (although
memory requirements of just putting the embedding model in the memory of a
personal computer can be prohibitive).

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


<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## fastText

### Setup

First install the `fastText` Python library by running in the terminal (within
the `dlnlp` environment):
```bash
pip install fasttext
```
The next step is to download a pre-trained model for a particular language,
preferably one of the [models officially distributed on the
website][fasttext-models].  The models are available in two formats: textual
and binary.  In practice the latter is easier to use in a PyTorch application.
In the following we use the official binary model for English, which can be
also downloaded from command line:
```bash
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
```
Note the file is large (4.2 GB) so its download can take some time.

### Basic usage

To test the setup, open the Python interpreter, import the `fasttext` library,
and try embedding some English words.

**WARNING**: The model is quite memory greedy and I wouldn't recommend trying
to use it if you haven't at least 16 GB of RAM on your personal computer.  An
alternative is to use a `fastText` model with [reduced
dimensionality][fasttext-reduce-dim].  **TODO**: Provide a link to one such
model for EN.

You can find the overview of the usage of the Python `fasttext` module
[here][fasttext-python-usage-overview].
```python
model = fasttext.load_model("cc.en.300.bin")
model['asparagus']
# => array([ 2.43494716e-02, -1.13698803e-02, -8.15532170e-03,  5.03604002e-02,
# =>                                        ...
# =>        -1.08037302e-02,  9.83848721e-02,  3.61668691e-02,  8.09734687e-03],
# =>       dtype=float32)
```
The output vector is a numpy array.
```python
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
which means that they won't be adapted during training of a PyTorch model using
them (which is most often the intended behavior, in contrast with the custom
embeddings we were using so far).

An important advantage of fastText embeddings in contrast with custom
word-level embeddings is that you no longer have to worry about spelling
mistakes or OOV words, thanks to fastText using sub-word information to produce
the embedding vectors (admittedly, using a character-level model to produce
word embeddings also alleviates this issue).
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
* Produce the fastText embeddings as part of the pre-processing
* Calculate the fastText embeddings as part of the model
In both cases, it is no longer necessary to encode the input words as integers
(since the fastText model takes strings on input).  The second strategy is the
only one that can be used if you want to adapt the embeddings (i.e. make them
parameters of the neural model).  Below we describe the implementation of the
first strategy.

#### Embedding during pre-processing




[fasttext-models]: https://fasttext.cc/docs/en/crawl-vectors.html#models "Official fastText models for 157 languages"
[fastext-python-usage-overview]: https://fasttext.cc/docs/en/python-module.html#usage-overview
[fasttext-reduce-dim]: https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
