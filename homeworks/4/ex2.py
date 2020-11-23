
import torch
from torch import Tensor
import torch.nn as nn

def calculate_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Calculate the cross-entropy loss between the predicted and
    the target tensors, in case where the target is a scalar value
    (tensor of dimension 0).

    Arguments:
    * pred: predicted (float) score tensor of shape C
    * target: scalar (int) target tensor
    where C is the number of output classes

    Examples:

    >>> pred = torch.FloatTensor([-10., 0., 10.])
    >>> target = torch.tensor(0)
    >>> isinstance(calculate_loss(pred, target), torch.Tensor)
    True
    >>> int(calculate_loss(pred, target))
    20
    >>> target = torch.tensor(1)
    >>> int(calculate_loss(pred, target))
    10
    >>> target = torch.tensor(2)
    >>> int(calculate_loss(pred, target))
    0
    """
    # Preliminary checks (optional)
    assert pred.dim() == 1      # vector
    assert target.dim() == 0    # scalar
    assert 0 <= target.item() < pred.shape[0]
    # TODO: Calculate and return the loss
    pass