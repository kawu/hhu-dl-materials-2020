import torch


def sumprod(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Calulate the "sum product" of two vectors of the same length.

    Examples:
    >>> x = torch.tensor([1, 1, 1])
    >>> y = torch.tensor([1, 1, 1])
    >>> assert sumprod(x, y) == 3
    >>> assert isinstance(sumprod(x, y), torch.Tensor)

    >>> x = torch.tensor([0, 1, 2])
    >>> y = torch.tensor([1, 1, 1])
    >>> assert sumprod(x, y) == 3

    >>> x = torch.tensor([0, 1, 2])
    >>> y = torch.tensor([2, 3, 4])
    >>> assert sumprod(x, y) == 11
    """
    # Assert v and w are vectors and have the same length
    assert len(v.shape) == len(w.shape) == 1
    assert v.shape[0] == w.shape[0]
    # TODO:
    pass
