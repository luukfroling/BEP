import numpy as np

def reshape2(a, dim1, dim2):
    """
    Reshapes the input array `a` into a new shape of (dim1, dim2).

    Parameters:
    - a: numpy array, input array with at least 3 dimensions.
    - dim1: int, first dimension of the reshaped array.
    - dim2: int, second dimension of the reshaped array.

    Returns:
    - a2: numpy array, reshaped array with shape (dim1, dim2).
    """

    nIter = dim1 // dim2
    a2 = np.zeros((dim1, dim2))

    print(dim1)
    print(dim2)

    for i in range(nIter):
        a2[i * dim2:(i + 1) * dim2, :] = a[:, :, i]

    return a2