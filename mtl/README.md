# Multi-task learning

The idea behind multi-task learning (MTL) is to design a model which tackles several
tasks in parallel.  We show here the application of this technique to joint POS
tagging and dependency parsing, taking the code from the [last session](https://github.com/kawu/hhu-dl-materials-2020/tree/main/char) (after an
important [bug fix](https://github.com/kawu/hhu-dl-materials-2020/commit/5d5b5b1902721a133ecb4df8b138870488c5943b))
as the starting point.


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
As another example, you can find the graphical representation of the first dependency tree in the training part of the English ParTUT treebank we use for our experiments [here](http://lindat.mff.cuni.cz/services/pmltq/#!/treebank/uden_partut22/query/IYWgdg9gJgpgBAbQGYQE4FsB+AiAIgSwGcAXVfAIwFdj8IxsBdAbiA/result/svg?filter=true&timeout=30&limit=100).

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
) -> List[Tuple[Tensor, EncOut]]:
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

We show here an implementation of a basic variant of the *biaffine* dependency
parsing model (see [here][biaffine-specs] for specs).  This model calculates a
score `score(x, y)` for each pair of words `(x, y)` in a given sentence and,
for each token `x` in the sentence, picks the head `y` which maximizes
`score(x, y)` (`y` can be the dummy root note, represented by `0`).
```python
class Biaffine(nn.Module):
    '''Calculate pairwise matching scores.

    Type: Tensor[N x D] -> Tensor[N x (N + 1)]

    For a given sequence (matrix) of word embeddings, calculate the matrix
    of pairwise matching scores.
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
**TODO**: add a link to an alternative implementation, with MLPs and bias
vector.

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
    """Calculate the accuracy of the model on the given dataset
    of (encoded) input/output pairs."""
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        # NOTE: Compare with the first element of the target pair
        correct += (y[1] == pred_y).sum()
        total += len(y[1])
    return float(correct) / total
```
and the loss:
```python
criterion = nn.CrossEntropyLoss()

def loss(pred: Tensor, gold: Tuple[Tensor, Tensor]) -> Tensor:
    return criterion(pred, gold[1])
```
and carry on to train the model.


[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
[biaffine-specs]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session12/u12_eng.pdf "Biaffine parser specification"
