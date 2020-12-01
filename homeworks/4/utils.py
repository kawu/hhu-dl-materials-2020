class Encoder:

    """Mapping between classes and the corresponding indices.

    >>> classes = ["English", "German", "French"]
    >>> enc = Encoder(classes)
    >>> assert "English" == enc.decode(enc.encode("English"))
    >>> assert "German" == enc.decode(enc.encode("German"))
    >>> assert "French" == enc.decode(enc.encode("French"))
    >>> set(range(3)) == set(enc.encode(cl) for cl in classes)
    True
    >>> for cl in classes:
    ...     ix = enc.encode(cl)
    ...     assert 0 <= ix <= enc.size()
    ...     assert cl == enc.decode(ix)
    """

    def __init__(self, classes):
        self.class_to_ix = {}
        self.ix_to_class = {}
        for cl in classes:
            if cl not in self.class_to_ix:
                ix = len(self.class_to_ix)
                self.class_to_ix[cl] = ix
                self.ix_to_class[ix] = cl

    def size(self):
        return len(self.class_to_ix)

    def encode(self, cl):
        return self.class_to_ix[cl]

    def decode(self, ix):
        return self.ix_to_class[ix]
