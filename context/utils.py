from typing import Iterable, TypeVar, Generic, Dict


T = TypeVar('T')
class Encoder(Generic[T]):

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

    def __init__(self, classes: Iterable[T]):
        self.class_to_ix: Dict[T, int] = {}
        self.ix_to_class: Dict[int, T] = {}
        for cl in classes:
            if cl not in self.class_to_ix:
                ix = len(self.class_to_ix)
                self.class_to_ix[cl] = ix
                self.ix_to_class[ix] = cl

    def size(self) -> int:
        return len(self.class_to_ix)

    def encode(self, cl: T) -> int:
        try:
            return self.class_to_ix[cl]
        except KeyError:
            return self.size()

    def decode(self, ix: int) -> T:
        return self.ix_to_class[ix]
