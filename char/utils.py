import torch
import torch.nn as nn

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


def train(
    model: nn.Module,
    train_data,
    dev_data,
    loss,
    accuracy,
    epoch_num=10,
    learning_rate=0.001,
    report_rate=10,
):
    """SGD training function."""
    # Use Adam to adapt the model's parameters
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for k in range(epoch_num):
        # Turn on the training mode
        model.train()
        # Variable to store the total loss on the training set
        total_loss = 0
        # Optional: use random dataset permutation in each epoch
        for i in torch.randperm(len(train_data)):
            x, y = train_data[i]
            z = loss(model(x), y)
            total_loss += z.item()
            z.backward()
            optim.step()
            optim.zero_grad()
        if k == 0 or (k+1) % report_rate == 0:
            with torch.no_grad():
                model.eval()
                train_acc = accuracy(model, train_data)
                dev_acc = accuracy(model, dev_data)
                print(f'@{k+1}: loss(train)={total_loss:.3f}, acc(train)={train_acc:.3f}, acc(dev)={dev_acc:.3f}')