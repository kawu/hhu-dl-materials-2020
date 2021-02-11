import torch
import torch.nn as nn

from torch.utils.data import DataLoader

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


def batch_loader(data_set, batch_size: bool, shuffle=False) -> DataLoader:
    """Create a batch data loader from the given data set.

    Using PyTorch Datasets and DataLoaders is especially useful when working
    with large datasets, which cannot be stored in the computer memory (RAM)
    all at once.

    Let's create a small dataset of numbers:
    >>> data_set = range(5)
    >>> for elem in data_set:
    ...     print(elem)
    0
    1
    2
    3
    4

    The DataLoader returned by the batch_loader function allows to
    process the dataset in batches.  For example, in batches of
    2 elements:
    >>> bl = batch_loader(data_set, batch_size=2, shuffle=False)
    >>> for batch in bl:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4]

    The last batch is of size 1 because the dataset has 5 elements in total.
    You can iterate over the dataset in batches over again:
    >>> for batch in bl:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4]

    For the sake of training of a PyTorch model, it may be better to shuffle
    the elements each time the stream of batches is created.
    To this end, use the `shuffle=True` option.
    >>> bl = batch_loader(data_set, batch_size=2, shuffle=True)

    DataLoader "visits" each element of the dataset once.
    >>> sum(len(batch) for batch in bl) == len(data_set)
    True
    >>> set(x for batch in bl for x in batch) == set(data_set)
    True
    """
    return DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=shuffle
    )


def train(
    model: nn.Module,
    train_data,
    dev_data,
    loss,
    accuracy,
    epoch_num=10,
    learning_rate=0.001,
    report_rate=10,
    batch_size=16
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
        Function which takes two arguments -- a model and a batch
        of dataset pairs.  The model is trained to minimize the
        cumulative loss over the entire training set
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
    batch_size : int
        Size of the batches
    """
    # Use Adam to adapt the model's parameters
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Create a batch loader for the training set
    bl = batch_loader(train_data, batch_size=batch_size, shuffle=True)
    for k in range(epoch_num):
        # Turn on the training mode
        model.train()
        # Variable to store the total loss on the training set
        total_loss = 0
        # For each batch in the training set
        for batch in bl:
            # Calculate the loss on the batch
            z = loss(model, batch)
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