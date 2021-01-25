# Multi-task learning

The idea behind multi-task learning (MTL) is to design a model which tackles
several related tasks in parallel.  We show here the application of this
technique to joint POS tagging and dependency parsing, taking the code from the
[last session](https://github.com/kawu/hhu-dl-materials-2020/tree/main/char)
(after an important [bug
fix](https://github.com/kawu/hhu-dl-materials-2020/commit/5d5b5b1902721a133ecb4df8b138870488c5943b))
as the starting point.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Dependency parsing](#dependency-parsing)
  - [Types and encoding](#types-and-encoding)
  - [Biaffine model](#biaffine-model)
- [Joint model](#joint-model)
- [Processing original data](#processing-original-data)
- [Footnotes](#footnotes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Dependency parsing

Before we can apply MTL to POS tagging and dependency parsing, we need to
develop a model which deals with the dependency parsing task alone.  Only then
can we combine it with a POS tagger.

The [CoNLL-U][conllu] dataset we were using for POS tagging contains dependency
annotations, in particular *head* annotations: for each token in a sentence,
its dependency head is specified as a single integer.  For instance, in the
following sentence:
```
# text = Only the Thought Police mattered.
1	Only	only	ADV	RB	_	4	advmod	_	_
2	the	the	DET	DT	_	4	det	_	_
3	Thought	thought	PROPN	NNP	_	4	compound	_	_
4	Police	police	PROPN	NNP	_	5	nsubj	_	_
5	mattered	matter	VERB	VBD	_	0	root	_	_
6	.	.	PUNCT	.	_	5	punct	_	_
```
The dependency heads selected for the individual tokens are `4, 4, 4, 5, 0, 5`,
i.e. the head of `Only` is `Police`, the head of `Police` is `mattered`, etc.
The dummy root of the sentence is represented by `0`.
As another example, you can find the graphical representation of the first
dependency tree in the training part of the English ParTUT treebank (which we
use for our experiments)
[here](http://lindat.mff.cuni.cz/services/pmltq/#!/treebank/uden_partut22/query/IYWgdg9gJgpgBAbQGYQE4FsB+AiAIgSwGcAXVfAIwFdj8IxsBdAbiA/result/svg?filter=true&timeout=30&limit=100).

To each dependency arc, the corresponding dependency label is assigned
(`advmod`, `det`, `compound`, etc.).  We will not be concerned with dependency
labels here, only with dependency heads.

### Types and encoding

In `data.py`, we introduce a new type to represent dependency heads, roughly a
synonym to `int`:
```python
# Dependency head
Head = NewType('Head', int)
# Think of the definition above as a more type-safe variant of `Head = int`
```
Since we want to perform tagging and parsing in parallel, we extend information
about POS tags in the dataset with information about the heads.
```python
# Output: a list of (POS tag, dependency head) pairs
Out = List[Tuple[POS, Head]]
```
This also means we need to update the `extract` function accordingly
(**WARNING**: as you will see, we didn't account for one particularity of the
dataset -- contractions).
<!--
```python
def extract(token_list: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in token_list:
        # NOTE: ignoring contractions
        if head is not None:
            inp.append(tok["form"])
            out.append((tok["upos"], tok["head"]))
    return inp, out
```
-->

The encoded representation of the target output should include both POS tags
and depednency heads.  Let's capture that at the level of types, too:
```python
# Encoded input: list of tensors, one per word
EncInp = List[Tensor]

# Encoded output: pair (encoded POS tags, dependency heads)
EncOut = Tuple[Tensor, Tensor]
```

The next step is to update the encoding procedure.  Note that we do not need a
separate `Encoder` for dependency heads -- this is because the heads are
already represented as integers and have appropriate form for subsequent
PyTorch processing.
```python
def create_encoders(
    data: List[Tuple[Inp, Out]]
) -> Tuple[Encoder[Char], Encoder[POS]]:
    """Create encoders for input characters and POS tags."""
    # TODO
    pass

def encode_with(
    data: List[Tuple[Inp, Out]],
    char_enc: Encoder[Char],
    pos_enc: Encoder[POS]
) -> List[Tuple[EncInp, EncOut]]:
    """Encode a dataset using character and output POS tag encoders."""
    # TODO
    pass
```

At this point, we can extract and encode the dataset (in `session.py`) and make
sure everything is in order.
```python
# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
char_enc, pos_enc = create_encoders(train_data)

# Encode the train set
enc_train = encode_with(train_data, char_enc, pos_enc)
enc_dev = encode_with(dev_data, char_enc, pos_enc)
```

### Biaffine model

We show here an implementation of a basic variant<sup>[1](#footnote1)</sup> of
the [*biaffine* dependency parsing model][biaffine-paper] (see
[here][biaffine-specs] for specs).  This model calculates a score `score(x, y)`
for each pair of words `(x, y)` in a given sentence and, for each token `x` in
the sentence, picks the head `y` which maximizes `score(x, y)` (`y` can be
the dummy root note, represented by `0`).
```python
class Biaffine(nn.Module):
    '''Calculate pairwise matching scores.

    Type: Tensor[N x D] -> Tensor[N x (N + 1)]

    For a given sequence (matrix) of word embeddings, calculate the matrix of
    pairwise matching scores.
    '''

    def __init__(self, emb_size: int):
        super().__init__()
        self.depr = nn.Linear(emb_size, emb_size)
        self.hedr = nn.Linear(emb_size, emb_size)
        self.root = nn.Parameter(torch.randn(emb_size))

    def forward(self, xs: Tensor):
        deps = self.depr(xs)
        heds = torch.cat((self.root.view(1, -1), self.hedr(xs)))
        return deps @ heds.t()
```
Here's a graphical representation of the scores that the biaffine module may
produce (once trained) on an example sentence:
<p align="center">
  <img src="imgs/thought_police_dependency_scores.png?raw=true" alt="Dependency parsing scores"/>
</p>

The definition of the end-to-end dependency parsing model is then the same as
that of the POS tagging model based on character-level embeddings, with the
exception that the last component scores dependency arcs rather than POS tags.
```python
model = nn.Sequential(
    Map(nn.Sequential(
        nn.Embedding(char_enc.size()+1, 50, padding_idx=char_enc.size()),
        SimpleLSTM(50, 200),
        Apply(lambda xs: xs[-1]),
    )),
    SimpleBiLSTM(200, 200),
    Biaffine(200)
)
```
We can then adapt the implementation of the accuracy function:
```python
def dep_accuracy(model, data):
    """Calculate the head prediction accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        # NOTE: Use the first element of the target pair for comparison
        correct += (y[1] == pred_y).sum()
        total += len(y[1])
    return float(correct) / total
```
and the loss:
```python
def loss(pred: Tensor, gold: Tuple[Tensor, Tensor]) -> Tensor:
    criterion = nn.CrossEntropyLoss()
    return criterion(pred, gold[1])
```
and carry on to train the model.
```python
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.001, report_rate=1)
# => @1: loss(train)=2881.698, acc(train)=0.607, acc(dev)=0.602
# =>                            ...
# => @10: loss(train)=270.933, acc(train)=0.939, acc(dev)=0.717
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.0001, report_rate=1)
# => @1: loss(train)=161.488, acc(train)=0.982, acc(dev)=0.733
# =>                            ...
# => @10: loss(train)=2.678, acc(train)=1.000, acc(dev)=0.741
```


## Joint model

Combining the POS tagging model with the dependency parsing model is rather
straightforward, thanks to the fact that both models rely on the same
contextualized word embeddings.  Here's a simple way to achieve that:
```python
class Joint(nn.Module):
    """Joint POS tagging / dependency parsing module.

    Type: EncInp -> Tuple[PosScores, DepScores]

    where:

    * EncInp is an encoded input sentence (list of tensors)
    * PosScores is a Tensor[N x T] with POS-related scores,
      where T is the number of distinct POS tags.
    * DepScores is a Tensor[N x (N + 1)] of dependency-related scores,
      one score vector of size (N + 1) per word.

    The two components (POS tagging and dependency parsing) are based
    on a common contextualized embedding representation.
    """

    def __init__(self,
        char_enc: Encoder[Char],    # Encoder for input characters
        pos_enc: Encoder[POS],      # Encoder for POS tags
        emb_size: int,              # Embedding size
        hid_size: int               # Hidden size used in LSTMs
    ):
        super().__init__()

        # Keep encoding objects for future use
        self.char_enc = char_enc
        self.pos_enc = pos_enc

        # Common part of the model: embedding and LSTM contextualization
        self.embed = nn.Sequential(
            Map(nn.Sequential(
                nn.Embedding(char_enc.size()+1, emb_size, padding_idx=char_enc.size()),
                SimpleLSTM(emb_size, hid_size),
                Apply(lambda xs: xs[-1]),
            )),
            SimpleBiLSTM(hid_size, hid_size),
        )

        # Scoring module for the POS tagging task
        self.score_pos = nn.Linear(hid_size, pos_enc.size())

        # Biaffine dependency scoring module
        self.score_dep = Biaffine(hid_size)

    def forward(self, xs: EncInp) -> Tuple[Tensor, Tensor]:
        embs = self.embed(xs)
        return (self.score_pos(embs), self.score_dep(embs))

model = Joint(char_enc, pos_enc, emb_size=50, hid_size=200)
```
The joint model calculates a pair of tensors (see the `forward` method): the
POS scores and the dependency scores, respectively.
<!-- **TODO**: Add an alternative implementation. -->

We can now adapt the loss function so as to measure the quality of the model as
a simple additive combination of its performance on POS tags and dependency
heads:
```python
def loss(
    pred: Tuple[Tensor, Tensor],
    gold: Tuple[Tensor, Tensor]
) -> Tensor:
    criterion = nn.CrossEntropyLoss()
    return criterion(pred[0], gold[0]) + criterion(pred[1], gold[1])
```
as well as the accuracy functions for the two tasks:
```python
def pos_accuracy(model, data):
    """Calculate the POS tagging accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x)[0], dim=1)
        correct += (y[0] == pred_y).sum()
        total += len(y[0])
    return float(correct) / total

def dep_accuracy(model, data):
    """Calculate the head prediction accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x)[1], dim=1)
        correct += (y[1] == pred_y).sum()
        total += len(y[1])
    return float(correct) / total
```
**NOTE**: At this point we could refactor the code to calculate both types of
accuracies in a single pass, and the `train` function to report both of them.

That's all, we can now apply the training procedure to train our joint model:
```python
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.001, report_rate=1)
# => @1: loss(train)=4367.376, acc(train)=0.654, acc(dev)=0.633
# =>                            ...
# => @10: loss(train)=437.589, acc(train)=0.943, acc(dev)=0.760
train(model, enc_train, enc_dev, loss, dep_accuracy, epoch_num=10, learning_rate=0.0001, report_rate=1)
# => @1: loss(train)=222.822, acc(train)=0.984, acc(dev)=0.781
# =>                            ...
# => @10: loss(train)=17.223, acc(train)=0.999, acc(dev)=0.789
pos_accuracy(model, enc_dev)
# => 0.9232182218956649
```


## Processing original data

To make the joint model more user-friendly, we can extend the `Joint` class
with two methods for tagging and parsing data in the original form.  Let's
first extract the functionality of encoding the input sentence in a separate
function (possibly in `data.py`):
```python
def encode_input(sent: List[Word], enc: Encoder[Char]) -> List[Tensor]:
    """Encode an input sentence given a character encoder."""
    return [
        torch.tensor([enc.encode(Char(char)) for char in word])
        for word in sent
    ]
```
The tagging/parsing methods can be then implemented as:
```python
    @torch.no_grad()
    def tag(self, sent: List[Word]) -> List[POS]:
        """Tag a sentence with POS tags."""
        xs = encode_input(sent, self.char_enc)
        embs = self.embed(xs)
        scores = self.score_pos(embs)
        ys = torch.argmax(scores, dim=1)
        return [self.pos_enc.decode(y.item()) for y in ys]

    @torch.no_grad()
    def parse(self, sent: List[Word]) -> List[Head]:
        """Predicted a dependency head for each word in a sentence."""
        xs = encode_input(sent, self.char_enc)
        embs = self.embed(xs)
        scores = self.score_dep(embs)
        ys = torch.argmax(scores, dim=1)
        return [y.item() for y in ys]
```
The `parse` method could be further extended with the [Chu–Liu/Edmonds'
algorithm][cle] to ensure that the resulting dependencies form a tree.

Once the model is trained, we can use them as follows:
```python
>>> model.tag("Only the Thought Police mattered .".split())
['ADV', 'DET', 'NOUN', 'PROPN', 'VERB', 'PUNCT']
>>> model.parse("Only the Thought Police mattered .".split())
[5, 4, 5, 5, 0, 5]
```

<!--
**TODO**: Mention other ways of combining models, e.g. the RoundRobin trick?
-->

## Footnotes

<a name="footnote1">1</a>: Here is a more robust variant which includes a bias
factor of a word being a head, regardless of a dependent:
```python
class Biaffine(nn.Module):
    '''Calculate pairwise matching scores.

    Type: Tensor[N x D] -> Tensor[N x (N + 1)]

    For a given sequence (matrix) of word embeddings, calculate the matrix of
    pairwise matching scores.
    '''

    def __init__(self, emb_size: int):
        super().__init__()
        self.depr = nn.Linear(emb_size, emb_size)
        self.hedr = nn.Linear(emb_size, emb_size)
        self.root = nn.Parameter(torch.randn(emb_size))
        self.bias = nn.Parameter(torch.randn(emb_size))

    def forward(self, xs: Tensor):
        deps = self.depr(xs)
        heds = torch.cat((self.root.view(1, -1), self.hedr(xs))).t()
        return (deps @ heds) + (self.bias.view(1, -1) @ heds)
```
Multi-layered perceptrons are also often used for the `depr` and `hedr`
components instead of `nn.Linear`.


[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
[biaffine-specs]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session12/u12_eng.pdf "Biaffine parser specification"
[biaffine-paper]: https://arxiv.org/pdf/1611.01734.pdf "Biaffine dependency parser"
[cle]: https://en.wikipedia.org/wiki/Edmonds%27_algorithm "Chu–Liu/Edmonds' algorithm"
