# TODO: Implement the missing pieces in the "OOV Calculation" section below.

from typing import Tuple, List, NewType

import conllu

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
# OOV Calculation
##################################################

def collect_OOVs(train_data: List[Tuple[Inp, Out]], dev_data: List[Tuple[Inp, Out]]):
    """Determine the set of OOV words in `dev` with respect to `train`.

    Examples:

    # Train set
    >>> x1 = "A black cat crossed the street".split()
    >>> y1 = "DET ADJ NOUN VERB DET NOUN".split()
    >>> train_set = [(x1, y1)]

    # Dev set
    >>> x2 = "A cat chased the dog".split()
    >>> y2 = "DET NOUN VERB DET NOUN".split()
    >>> dev_set = [(x2, y2)]

    # OOV words in dev w.r.t. train
    >>> sorted(collect_OOVs(train_set, dev_set))
    ['chased', 'dog']
    """
    # TODO:
    pass

#################################################
# EVALUATION SECTION START: DO NOT MODIFY!
#################################################

train_set = parse_and_extract("UD_English-ParTUT/en_partut-ud-train.conllu")
dev_set = parse_and_extract("UD_English-ParTUT/en_partut-ud-dev.conllu")
oovs = collect_OOVs(train_set, dev_set)
if len(oovs) == 311:
    print(f"OK: found {len(oovs)} OOV words in the dev set, as expected")
else:
    print(f"FAILED: found {len(oovs)} OOV words in the dev set, in lieu of 311")

# test_set = parse_and_extract("UD_English-ParTUT/en_partut-ud-test.conllu")
# test_oovs = collect_OOVs(train_set, test_set)
# print(f"Found {len(test_oovs)} OOV words in the test set")

#################################################
# EVALUATION SECTION END
#################################################