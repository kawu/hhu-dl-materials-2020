
from typing import List, Tuple

sample_raw_data = """German: Bach, Engel, Gottlieb, Zimmermann
English: Alderson, Churchill, Ecclestone
Czech: Blazejovsky, Hruskova, Veverka
Greek: Antonopoulos, Leontarakis
Japanese: Fujishima, Hayashi
Korean: Park, Seok
Spanish: Álvarez, Pérez
English: Keighley, Reynolds
"""

# Input: person name (Bach, Reynolds, etc.)
Name = str

# Output: the language (German, English, etc.)
Lang = str

def parse(raw_data: str) -> List[Tuple[Name, Lang]]:
    """Extract the name/language pairs from a raw string.

    The raw input string consists of several lines and each line has the format:

        <lang>: <name1>, <name2>, <name3>, ...

    Several lines with the same language can be provided.  The same name can be
    specified for a given language several times (in which case the duplicates
    should be preserved).  One name can be also assigned to different languages.
    Otherwise, the order in which the name/language pairs are returned is not
    specified.  See also `sample_raw_data` above.

    Examples:

    >>> data_set = parse('German: Bach, Engel')
    >>> print(sorted(data_set))
    [('Bach', 'German'), ('Engel', 'German')]

    >>> data_set = parse('German: Bach, Engel\\nEnglish: Alderson, Churchill')
    >>> print(sorted(data_set))
    [('Alderson', 'English'), ('Bach', 'German'), ('Churchill', 'English'), ('Engel', 'German')]

    # Duplicates should be preserved
    >>> data_set1 = parse('German: Bach, Bach')
    >>> len(data_set1)
    2
    >>> data_set2 = parse('German: Bach\\nGerman: Bach')
    >>> sorted(data_set1) == sorted(data_set2)
    True

    >>> data_set = parse(sample_raw_data)
    >>> print(len(data_set))
    20

    # Retrive Enlish names only
    >>> en_names = [name for name, lang in data_set if lang == 'English']
    >>> print(sorted(en_names))
    ['Alderson', 'Churchill', 'Ecclestone', 'Keighley', 'Reynolds']
    """
    # TODO:
    pass
