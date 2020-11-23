from typing import Tuple, List

import torch
from torch import Tensor

from utils import Encoder

##################################################
# Data
##################################################

# Input: person name (Bach, Reynolds, etc.)
Name = str

# Output: the language (German, English, etc.)
Lang = str

# Name language prediction dataset
data: List[Tuple[Name, Lang]] = [
    ('Bach', 'German'),
    ('Engel', 'German'),
    ('Gottlieb', 'German'),
    ('Zimmermann', 'German'),
    ('Alderson', 'English'),
    ('Churchill', 'English'),
    ('Ecclestone', 'English'),
    ('Keighley', 'English'),
    ('Reynolds', 'English'),
    ('Blazejovsky', 'Czech'),
    ('Hruskova', 'Czech'),
    ('Veverka', 'Czech'),
    ('Antonopoulos', 'Greek'),
    ('Leontarakis', 'Greek'),
    ('Fujishima', 'Japanese'),
    ('Hayashi', 'Japanese'),
    ('Park', 'Korean'),
    ('Seok', 'Korean'),
    ('Álvarez', 'Spanish'),
    ('Pérez', 'Spanish'),
]

# Encoded dataset: a list of (input, output) pairs encoded as tensors.  The
# list is populated below.
enc_data: List[Tuple[Tensor, Tensor]] = []

##################################################
# Encoding
##################################################

# Create the encoder for the input characters
char_enc = Encoder(char for name, _ in data for char in name)
# Create the encoder for the output languages
lang_enc = Encoder(lang for _, lang in data)

# Encode the dataset
for name, lang in data:
    enc_inp = torch.tensor([char_enc.encode(char) for char in name])
    enc_out = torch.tensor(lang_enc.encode(lang))
    enc_data.append((enc_inp, enc_out))
