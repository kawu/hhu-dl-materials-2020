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
# Pre-processing
##################################################

# def preprocess(inp: Inp) -> Inp:
#     """Lower-case all words in the input sentence."""
#     return [x.lower() for x in inp]
# 
# # Apply the pre-processing function to the dataset
# for i in range(len(data)):
#     inp, out = data[i]
#     data[i] = preprocess(inp), out

##################################################
# Encoding
##################################################


def create_encoders(data: List[Tuple[Inp, Out]]) \
        -> Tuple[Encoder[Word], Encoder[POS]]:
    word_enc = Encoder(word for inp, _ in data for word in inp)
    pos_enc = Encoder(pos for _, out in data for pos in out)
    return (word_enc, pos_enc)


def encode_with(
    data: List[Tuple[Inp, Out]],
    word_enc: Encoder[Word],
    pos_enc: Encoder[POS]
) -> List[Tuple[Tensor, Tensor]]:
    # An internal function to handle encoding exceptions
    def encode_word(word):
        try:
            return word_enc.encode(word)
        except KeyError:
            return word_enc.size()

    enc_data = []
    for inp, out in data:
        enc_inp = torch.tensor([encode_word(word) for word in inp])
        enc_out = torch.tensor([pos_enc.encode(pos) for pos in out])
        enc_data.append((enc_inp, enc_out))
    return enc_data


# class EncData(TypedDict):
#     '''
#     Encoded dataset, together with the corresponding
#     encoders for the words and the POS tags
#     '''
#     data: List[Tuple[Tensor, Tensor]]
#     word_enc: Encoder[Word]
#     pos_enc: Encoder[POS]

# def encode(data: List[Tuple[Inp, Out]]) -> EncData:
#     """Encode a dataset as tensors."""
#     word_enc = Encoder(word for inp, _ in data for word in inp)
#     pos_enc = Encoder(pos for _, out in data for pos in out)
#     enc_data = []
#     for inp, out in data:
#         enc_inp = torch.tensor([word_enc.encode(word) for word in inp])
#         enc_out = torch.tensor([pos_enc.encode(pos) for pos in out])
#         enc_data.append((enc_inp, enc_out))
#     return EncData(
#         data=enc_data,
#         word_enc=word_enc,
#         pos_enc=pos_enc
#     )
