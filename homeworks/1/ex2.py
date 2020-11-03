import torch


def ata(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Apply matrix `M` to each word vector in matrix `w`.

    Examples:
    >>> w = torch.tensor([[1, 2, 3], [2, 3, 1]], dtype=torch.float)
    >>> M = torch.zeros(3, 3)
    >>> ata(M, w)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    >>> M = torch.ones(3, 3)
    >>> torch.mv(M, w[0])
    tensor([6., 6., 6.])
    >>> torch.mv(M, w[1])
    tensor([6., 6., 6.])
    >>> ata(M, w)
    tensor([[6., 6., 6.],
            [6., 6., 6.]])

    >>> M = torch.eye(3, dtype=torch.float)
    >>> assert (ata(M, w) == w).all()

    >>> M = torch.randn(3, 3)
    >>> for i in range(2):
    ...     x = torch.mv(M, w[i])
    ...     y = ata(M, w)[i]
    ...     # Allow for numerical errors (just in case)
    ...     assert (abs(x - y) < 1e-5).all()
    """
    # Make sure the shapes match
    assert M.dim() == w.dim() == 2
    assert M.shape[1] == w.shape[1]
    # TODO:
    pass
