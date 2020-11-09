from typing import List, Tuple

import torch
from torch import Tensor

from utils import Encoder

# Input: person name (Bach, Reynolds, etc.)
Name = str

# Output: the language (German, English, etc.)
Lang = str

# Name language prediction dataset
sample_data_set: List[Tuple[Name, Lang]] = [
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

def encode_data(data_set: List[Tuple[Name, Lang]]) \
        -> List[Tuple[Tensor, Tensor]]:
    """Encode a dataset of name/language pairs with tensors.

    See `sample_data_set` above for an example of a non-encoded dataset.

    Examples:

    >>> data = [('Bach', 'German'), ('Mann', 'German'), ('Miles', 'English')]
    >>> enc_data = encode_data(data)
    
    The `enc_data` object should at this point look like this:
        
        [ (tensor([0, 1, 2, 3]), tensor(0))
        , (tensor([4, 1, 5, 5]), tensor(0))
        , (tensor([4, 6, 7, 8, 9]), tensor(1))
        ]
    
    although the exact values inside may differ (e.g. you may use
    index `0` to represent character `a`)

    >>> assert len(enc_data) == len(data)
    >>> for (inp, out), (x, y) in zip(data, enc_data):
    ...    assert len(inp) == len(x)
    ...    assert isinstance(x, Tensor)
    ...    assert isinstance(y, Tensor)

    # There are 10 distinct names in the dataset
    >>> set(ix.item() for x, y in enc_data for ix in x)
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # And 2 distinct languages
    >>> set(y.item() for x, y in enc_data)
    {0, 1}

    # Finally, some tests with the sample data set
    >>> enc_data = encode_data(sample_data_set)
    >>> set(y.item() for x, y in enc_data)
    {0, 1, 2, 3, 4, 5, 6}
    >>> set(ix.item() for x, y in enc_data for ix in x) # doctest:+ELLIPSIS
    {0, 1, ..., 36, 37}

    # The number of occurrences of the most frequent character
    >>> from collections import Counter
    >>> cnt = Counter(char for name, lang in sample_data_set for char in name)
    >>> cnt.most_common(1)[0]
    ('e', 16)
    >>> cnt = Counter(ix.item() for x, y in enc_data for ix in x)
    >>> cnt.most_common(1)[0][1]
    16
    """
    # TODO:
    pass
