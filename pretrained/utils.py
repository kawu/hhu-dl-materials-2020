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
    """Training function based on stochastic gradient descent.

    Parameters
    ----------
    model : nn.Module
        PyTorch module to train
    train_data : list
        List of encoded input/output pairs; the model is trained
        to minimize the loss (see the `loss` parameter) over this
        dataset
    dev_data : list
        List of encoded input/output pairs used for development;
        The function reports the accuracy of the model on the dev_data
        every couple of epochs (see `report_rate`)
    loss : function
        Function which takes two arguments -- the output of the model
        on a given input, and the target output (both in encoded form)
        -- and returns a float scalar tensor.  The model is trained
        to minimize the cumulative loss over the entire training set
        (see `train_data`).
    accuracy : function
        Helper function with two arguments -- a PyTorch model and a
        dataset -- which calculates the accuracy of the model.
        Used exclusively for reporting (see also `report_rate`).
    epoch_num : int
        The number of epochs, i.e., the number of passes of the algorithm
        over the entire training set (`train_data`).
    learning_rate : float
        Learning rate, determines the size of the step in each iteration
        of the algorithm.
    report_rate : int
        Determines the frequency of reporting the `loss` on the
        training set (`train_data`) and accuracy on both datasets
        (`train_data` and `dev_data`)
    """
    # Use Adam to adapt the model's parameters
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for k in range(epoch_num):
        # Turn on the training mode
        model.train()
        # Variable to store the total loss on the training set
        total_loss = 0
        # Optional: use random dataset permutation in each epoch
        for i in torch.randperm(len(train_data)):
            # Pick a dataset pair on position i
            x, y = train_data[i]
            # Calculate the loss between the output of the model
            # and the target output
            z = loss(model(x), y)
            # Update the total loss on the training set (used
            # for reporting)
            total_loss += z.item()
            # Calculate the gradients using backpropagation
            z.backward()
            # Update the parameters along the gradients
            optim.step()
            # Zero-out the gradients
            optim.zero_grad()
        # Report the loss and accuracy values after the first epoch
        # and every `report_rate` epochs
        if k == 0 or (k+1) % report_rate == 0:
            # No need to calculate the gradients during evaluation
            with torch.no_grad():
                # Enter evaluation mode, important in case dropout is used
                model.eval()
                train_acc = accuracy(model, train_data)
                dev_acc = accuracy(model, dev_data)
                print(f'@{k+1}: loss(train)={total_loss:.3f}, acc(train)={train_acc:.3f}, acc(dev)={dev_acc:.3f}')