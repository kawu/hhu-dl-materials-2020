from typing import Tuple, List, TypedDict, NewType

import torch
from torch import Tensor

import conllu

from utils import Encoder

##################################################
# Types
##################################################

# Word form
Word = NewType('Word', str)

# Single character
Char = NewType('Char', str)

# POS tag
POS = NewType('POS', str)

# Dependency head
Head = NewType('Head', int)

# Input: a list of words
Inp = List[Word]

# Output: a list of (POS tag, dependency head) pairs
Out = List[Tuple[POS, Head]]

# Encoded input: list of tensors, one per word
EncInp = List[Tensor]

# Encoded output: pair (encoded POS tags, dependency heads)
EncOut = Tuple[Tensor, Tensor]

##################################################
# Parsing and extraction
##################################################

def extract(token_list: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in token_list:
        form: Word = tok["form"]
        upos: POS = tok["upos"]
        head: Head = tok["head"]
        # Ignore contractions (we assume tokenization is solved)
        if head is not None:
            inp.append(form)
            out.append((upos, head))
    return inp, out

def parse_and_extract(conllu_path) -> List[Tuple[Inp, Out]]:
    """Parse a CoNLL-U file and return the list of input/output pairs."""
    data = []
    with open(conllu_path, "r", encoding="utf-8") as data_file:
        for token_list in conllu.parse_incr(data_file):  # type: ignore
            data.append(extract(token_list))
    return data

##################################################
# Encoding
##################################################

def create_encoders(
    data: List[Tuple[Inp, Out]]
) -> Tuple[Encoder[Char], Encoder[POS]]:
    """Create a pair of encoders, for words and POS tags respectively.

    Parameters
    ----------
    data : List[Tuple[Inp, Out]]
        List of input/output pairs based on which the encoders
        will be created; this parameter should only contain the
        training pairs, and not development or evaluation pairs.

    Returns
    -------
    (char_enc, pos_enc) : Tuple[Encoder[Char], Encoder[POS]]
        Pair of encoders for input characters and output POS tags.
    """
    # Enumerate all input characters present in the dataset
    # and create the encoder out of the resulting iterable
    char_enc = Encoder(
        Char(char)
        for inp, _ in data
        for word in inp
        for char in word
    )
    # Enumerate all POS tags in the dataset and create
    # the corresponding encoder
    pos_enc = Encoder(pos for _, out in data for pos, head in out)
    return (char_enc, pos_enc)

def encode_input(sent: Inp, char_enc: Encoder[Char]) -> EncInp:
    """Encode input sentence given a character encoder."""
    return [
        torch.tensor([char_enc.encode(Char(char)) for char in word])
        for word in sent
    ]

def encode_output(out: Out, pos_enc: Encoder[POS]) -> EncOut:
    """Encode output pair given a POS encoder."""
    enc_pos = torch.tensor([pos_enc.encode(pos) for pos, head in out])
    enc_head = torch.tensor([head for pos, head in out])
    return (enc_pos, enc_head)

def encode_with(
    data: List[Tuple[Inp, Out]],
    char_enc: Encoder[Char],
    pos_enc: Encoder[POS]
) -> List[Tuple[EncInp, EncOut]]:
    """Encode a dataset using given input character and output POS tag encoders.

    Parameters
    ----------
    data : List[Tuple[Inp, Out]]
        List of input/output pairs to encode
    char_enc : Encoder[Char]
        Encoder able to encode (as integers) input characters
    pos_enc : Encoder[POS]
        Encoder able to encode (as integers) output POS tags

    Returns
    -------
    enc_data : List[Tuple[EncInp, EncOut]]
        List of encoded input/output pairs

    If there are no unknown (OOV) symbols in the `data` parmeter, then
    the resulting encoded dataset `enc_data` is isomorphic to `data`,
    i.e. it can be decoded back to `data` using the `decode` method
    of the `char_enc` and `pos_enc` encoders.
    """
    enc_data = []
    for inp, out in data:
        enc_inp = encode_input(inp, char_enc)
        enc_out = encode_output(out, pos_enc)
        enc_data.append((enc_inp, enc_out))
    return enc_data
