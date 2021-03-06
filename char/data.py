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

# Input: a list of words
Inp = List[Word]

# Output: a list of POS tags
Out = List[POS]

##################################################
# Parsing and extraction
##################################################

def extract(token_list: conllu.TokenList) -> Tuple[Inp, Out]:
    """Extract the input/output pair from a CoNLL-U sentence."""
    inp, out = [], []
    for tok in token_list:
        inp.append(tok["form"])
        out.append(tok["upos"])
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
        char
        for inp, _ in data
        for word in inp
        for char in word
    )
    # Enumerate all POS tags in the dataset and create
    # the corresponding encoder
    pos_enc = Encoder(pos for _, out in data for pos in out)
    return (char_enc, pos_enc)

def encode_with(
    data: List[Tuple[Inp, Out]],
    char_enc: Encoder[Char],
    pos_enc: Encoder[POS]
) -> List[Tuple[List[Tensor], Tensor]]:
    """Encode a dataset using given input word and output POS tag encoders.

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
    enc_data : List[Tuple[List[Tensor], Tensor]]
        List of encoded input/output pairs

    If there are no unknown (OOV) symbols in the `data` parmeter, then
    the resulting encoded dataset `enc_data` is isomorphic to `data`,
    i.e. it can be decoded back to `data` using the `decode` method
    of the `char_enc` and `pos_enc` encoders.
    """
    enc_data = []
    for inp, out in data:
        enc_inp = [
            torch.tensor([char_enc.encode(char) for char in word])
            for word in inp
        ]
        enc_out = torch.tensor([pos_enc.encode(pos) for pos in out])
        enc_data.append((enc_inp, enc_out))
    return enc_data
