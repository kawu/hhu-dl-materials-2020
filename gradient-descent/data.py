from typing import Tuple, List

import torch
from torch import Tensor

import conllu

from utils import Encoder

##################################################
# Data
##################################################

# Raw dataset, stored as string
raw_data: str = """
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

# Dataset: a list of non-encoded (input, output) pairs.  The list is populated
# and pre-processed below.
data: List[Tuple[Inp, Out]] = []

# Encoded dataset: a list of (input, output) pairs encoded as tensors.  The
# list is populated below.
enc_data: List[Tuple[Tensor, Tensor]] = []

##################################################
# Extraction
##################################################

def extract(token_list: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in token_list:
        inp.append(tok["form"])
        out.append(tok["upos"])
    return inp, out

for token_list in conllu.parse(raw_data):
    data.append(extract(token_list))

##################################################
# Pre-processing
##################################################

def preprocess(inp: Inp) -> Inp:
    """Lower-case all words in the input sentence."""
    return [x.lower() for x in inp]

# Apply the pre-processing function to the dataset
for i in range(len(data)):
    inp, out = data[i]
    data[i] = preprocess(inp), out

##################################################
# Encoding
##################################################

# Create the encoder fo the input words
word_enc = Encoder(word for inp, _ in data for word in inp)
# Create the encoder for the POS tags
tag_enc = Encoder(pos for _, out in data for pos in out)

for inp, out in data:
    enc_inp = torch.tensor([word_enc.encode(word) for word in inp])
    enc_out = torch.tensor([tag_enc.encode(pos) for pos in out])
    enc_data.append((enc_inp, enc_out))
