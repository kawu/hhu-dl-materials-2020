import torch
from torch import tensor
import torch.nn.functional as F


def accuracy(model, data):
    """Accuracy of the model on a dataset of encoded input/output pairs.

    Arguments:
    * model: a function (typically neural module) which, given an input
        tensor, returns a vector of scores
    * data: list of encoded input/output pairs, where input has the shape
        consistent with the input required by the model, and output is
        a scalar tensor (single value), which represents the target class

    Examples:
    
    # Sample dataset
    >>> data = [
    ...     (tensor([0, 1, 2]), tensor(0)),
    ...     (tensor([1, 2, 3]), tensor(1)),
    ...     (tensor([5, 2]), tensor(2)),
    ... ]

    # Let's create a dummy model which always returns the highest score
    # for index 0
    >>> model = lambda _: tensor([1., 0., -1.])

    # Calculate the accuracy
    >>> accuracy(model, data) == 1. / 3.
    True

    # Another model, which provides the highest score for the index
    # corresponding to the smallest input value
    >>> model = lambda xs: F.one_hot(min(xs), num_classes=3)
    >>> model(tensor([0, 1, 2]))
    tensor([1, 0, 0])
    >>> model(tensor([2, 2, 2]))
    tensor([0, 0, 1])

    # .. and its accuracy on the same dataset
    >>> accuracy(model, data) == 3. / 3.
    True
    """
    pass
