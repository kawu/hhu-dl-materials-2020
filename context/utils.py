from typing import Iterable, TypeVar, Generic, Dict

import torch


T = TypeVar('T')
class Encoder(Generic[T]):

    """Mapping between classes and the corresponding indices.

    >>> classes = ["English", "German", "French"]
    >>> enc = Encoding(classes)
    >>> assert "English" == enc.decode(enc.encode("English"))
    >>> assert "German" == enc.decode(enc.encode("German"))
    >>> assert "French" == enc.decode(enc.encode("French"))
    >>> set(range(3)) == set(enc.encode(cl) for cl in classes)
    True
    >>> for cl in classes:
    ...     ix = enc.encode(cl)
    ...     assert 0 <= ix <= enc.class_num
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
        return self.class_to_ix[cl]

    def decode(self, ix: int) -> T:
        return self.ix_to_class[ix]


# Accuracy of the model
# TODO: Remove from the initial session code
def accuracy(model, data):
    correct, total = 0, 0
    for x, y in data:
        pred_y = torch.argmax(model(x), dim=1)
        correct += (y == pred_y).long().sum()
        total += len(y)
    return float(correct) / total


# TODO: Remove from the initial session code
def train(model, loss, train_data, dev_data, epoch_num=10, learning_rate=0.001, report_rate=1):
    # Create an optimizer to adapt the model's parameters
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Perform SGD for 1000 epochs
    for k in range(epoch_num):
        # Put the model in the training mode at the beginning of each epoch
        model.train()
        total_loss: float = 0.0
        for i in torch.randperm(len(train_data)):
            x, y = train_data[i]
            z = loss(model(x), y)
            total_loss += z.item()
            z.backward()
            optim.step()
        if k == 0 or (k+1) % report_rate == 0:
            # Switch off gradient evaluation
            with torch.no_grad():
                model.eval() # Put the model in the evaluation mode
                acc_train = accuracy(model, train_data)
                acc_dev = accuracy(model, dev_data)
                print(
                    f'@{k+1}: loss(train)={total_loss:.3f}, '
                    f'acc(train)={acc_train:.3f}, '
                    f'acc(dev)={acc_dev:.3f}'
                )
