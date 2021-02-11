from typing import Tuple, List, TypedDict, NewType

import torch
from torch import Tensor

import fasttext     # type: ignore
from bert_serving.client import BertClient  # type: ignore

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
EncInp = Tensor

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

# def encode_input(sent: Inp, ft_model) -> EncInp:
#     """Embed an input sentence given a fastText model."""
#     return torch.tensor([
#         ft_model[word]
#         for word in sent
#     ])

def encode_input(sent: Inp, bc: BertClient, device) -> EncInp:
    """Embed an input sentence given a BERT client.

    **NOTE**: The function assumes an uncased BERT model.
    """
    # Lower-case input for an uncased BERT model
    lower_sent = [x.lower() for x in sent]
    # Retrieve the embeddings of the sentence
    xs = torch.tensor(bc.encode([lower_sent], is_tokenized=True).copy()).to(device).squeeze(0)
    # Discard [CLS] and [SEP] embeddings
    xs = xs[1:len(sent)+1]
    # Make sure the lenghts match, just in case
    assert len(xs) == len(sent)
    return xs

def encode_output(out: Out, pos_enc: Encoder[POS], device) -> EncOut:
    """Encode output pair given a POS encoder."""
    enc_pos = torch.tensor([pos_enc.encode(pos) for pos, head in out], device=device)
    enc_head = torch.tensor([head for pos, head in out], device=device)
    return (enc_pos, enc_head)

def encode_with(
    data: List[Tuple[Inp, Out]],
    bert_client,
    pos_enc: Encoder[POS],
    device
) -> List[Tuple[EncInp, EncOut]]:
    """Encode a dataset using given input BERT client and
    output POS tag encoder.

    Parameters
    ----------
    data : List[Tuple[Inp, Out]]
        List of input/output pairs to encode/embed
    bert_client : BertClient
        BERT client for embedding input words
    pos_enc : Encoder[POS]
        Encoder able to encode (as integers) output POS tags
    device
        Target device ('cpu', 'cuda')

    Returns
    -------
    enc_data : List[Tuple[EncInp, EncOut]]
        List of embedded/encoded input/output pairs
    """
    enc_data = []
    for inp, out in data:
        enc_inp = encode_input(inp, bert_client, device)
        enc_out = encode_output(out, pos_enc, device)
        enc_data.append((enc_inp, enc_out))
    return enc_data
