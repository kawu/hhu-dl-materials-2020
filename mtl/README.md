# Multi-task learning

The idea behind multi-task learning (MTL) is to design a model which tackles several
tasks in parallel.  We show here the application of this technique to joint POS
tagging and dependency parsing.


## Dependency parsing

Before we can apply MTL to POS tagging and dependency parsing, we need to
develop a model which deals with the dependency parsing tasks alone.  Only then
we will combine it with a POS tagger.

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

To each dependency arc, the corresponding dependency label is assigned
(`advmod`, `det`, `compound`, etc.).  We will not be concerned with dependency
labels here, only with dependency heads.

### Types and encoding

In `data.py`, we introduce a new type to represent dependency heads, roughly a
synonym to `int`:
```python
# Dependency head
Head = NewType('Head', int)
```
Since we want to perform tagging and parsing in parallel, we extend information
about POS tags in the dataset with information about the heads.
```python
# Output: a list of (POS tag, dependency head) pairs
Out = List[Tuple[POS, Head]]
```
This also means we need to update the `extract` function accordingly.
<!--
```python
def extract(token_list: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in token_list:
        upos = tok["upos"]
        head = tok["head"]
        # NOTE: ignoring contractions
        if head is not None:
            inp.append(tok["form"])
            out.append((upos, head))
    return inp, out
```
-->

The encoded representation of the target output should include both POS tags
and depednency heads.  We can use a `@dataclass` to capture both, called
`EncOut` (standing for "Enc(oded) Out(put)").
```python
@dataclass
class EncOut:
    pos: Tensor     # Target POS tags (encoded)
    dep: Tensor     # Target dependency heads
```

The next step is to update the encoding procedure.  Will will encode (and,
consequently, embed) entire input words for simplicity.  Note that we do not
need a separate `Encoder` for dependency heads -- this is because the heads are
already represented as integers and have appropriate form for subsequent
PyTorch processing.
```python
def create_encoders(
    data: List[Tuple[Inp, Out]]
) -> Tuple[Encoder[Word], Encoder[POS]]:
    """Create a pair of encoders, for input words and POS tags respectively."""
    word_enc = Encoder(word for inp, _ in data for word in inp)
    pos_enc = Encoder(pos for _, out in data for pos, _head in out)
    return (word_enc, pos_enc)

def encode_with(
    data: List[Tuple[Inp, Out]],
    word_enc: Encoder[Word],
    pos_enc: Encoder[POS]
) -> List[Tuple[Tensor, EncOut]]:
    """Encode a dataset using given input word and output POS tag encoders."""
    enc_data = []
    for inp, out in data:
        enc_inp = torch.tensor([word_enc.encode(word) for word in inp])
        enc_pos = torch.tensor([pos_enc.encode(pos) for pos, _ in out])
        enc_dep = torch.tensor([head for _, head in out])
        enc_data.append((enc_inp, EncOut(enc_pos, enc_dep)))
    return enc_data
```

At this point, we can extract and encode the dataset (in `session.py`) and make
sure everything is in order. (**TODO**: inp\_enc -> char\_enc?)
```python
# Parse the training data and create the character/POS encoders
train_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_data = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
inp_enc, pos_enc = create_encoders(train_data)

# Encode the train set
enc_train = encode_with(train_data, inp_enc, pos_enc)
enc_dev = encode_with(dev_data, inp_enc, pos_enc)
```

### Biaffine model

We show here an implementation of a basic variant of the *biaffine* dependency
parsing model (TODO: add link to the paper).  This model calculates a score
`score(x, y)` for each pair of words `(x, y)` in a given sentence and, for each
token `x` in the sentence, picks the head `y` which maximizes `score(x, y)`
(`y` can be the dummy root note, represented by `0`).
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
TODO: the entire model.
The definition of the end-to-end dependency parsing model is then:
```python
model = nn.Sequential(
    nn.Embedding(inp_enc.size()+1, emb_size, padding_idx=inp_enc.size()),
    SimpleLSTM(emb_size, hid_size),
    Biaffine(hid_size)
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
        correct += (y.dep == pred_y).sum()
        total += len(y.dep)
    return float(correct) / total
```
and the loss:
```python
criterion = nn.CrossEntropyLoss()

def loss(pred: Tensor, gold: EncOut) -> Tensor:
    return criterion(pred, gold.dep)
```
and carry on to train the model.




[conllu]: https://universaldependencies.org/format.html "CoNLL-U format"
