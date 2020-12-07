from typing import Tuple, List, NewType

import csv

import torch
from torch import Tensor

from utils import Encoder

##################################################
# Types
##################################################

# Input: person name (Bach, Reynolds, etc.)
Name = str

# Input character
Char = str

# Output: the language (German, English, etc.)
Lang = str

##################################################
# Data extraction
##################################################

def load_data(file_path: str) -> List[Tuple[Name, Lang]]:
    """Load the dataset from a .csv file."""
    data_set = []
    with open(file_path, 'r', encoding='utf8') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for name, lang in csv_reader:
            data_set.append((name, lang))
    return data_set

##################################################
# Encoding
##################################################

def create_encoders(data: List[Tuple[Name, Lang]]) \
        -> Tuple[Encoder[Char], Encoder[Lang]]:
    """Create the encoders for the input characters and
    the output languages."""
    char_enc = Encoder(char for name, lang in data for char in name)
    lang_enc = Encoder(lang for name, lang in data)
    return char_enc, lang_enc

def encode_with(
    data: List[Tuple[Name, Lang]],
    char_enc: Encoder[Char],
    lang_enc: Encoder[Lang],
) -> List[Tuple[Tensor, Tensor]]:
    enc_data = []
    for name, lang in data:
        enc_name = torch.tensor([char_enc.encode(char) for char in name])
        enc_lang = torch.tensor(lang_enc.encode(lang))
        enc_data.append((enc_name, enc_lang))
    return enc_data
