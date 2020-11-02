# Tensors

This document covers the basic tensor operations you should get familiar with.

Tensors can be understood as multi-dimensional arrays (or matrices) containing
values of a certain type (integer, float, bool).
* 0-order tensor is a single value 
* 1-order tensor is a vector of values (one dimensional array)
* 2-order tensor is a matrix of values (two dimensional array)
and so on.

You can also go through the lengthier but arguably more comprehensive PyTorch
[documentation on tensors](https://pytorch.org/docs/stable/tensors.html).

## Creation

One way to create a tensor is to directly provide its value as argument to
`torch.tensor`:
```python
# First import the torch module
import torch

# Create a tensor which contains a single value `13`
value_13 = torch.tensor(13)
value_13              # => tensor(13)

# Create a vector with the values in range(1, 10)
vector_1_to_10 = torch.tensor(range(1, 10))
vector_1_to_10        # => tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create a 3x3 matrix with the same values
matrix_1_to_10 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_1_to_10        # => tensor([[1, 2, 3],
                      #            [4, 5, 6],
                      #            [7, 8, 9]])
```

<!--
You can also create tensors of higher dimensions (e.g. ,,cubes'' of values),
but we will not need this functionality today.
-->

**WARNING**: do not use `torch.tensor` to combine existing tensors, only to
create new ones.
<!---
Otherwise, [backpropagation](#backpropagation) may not work.
-->

The second way to create a tensor is to use one of the many functions which
take as argument the desired shape of the tensor, such as:
* [zeros](https://pytorch.org/docs/1.6.0/generated/torch.zeros.html?highlight=zeros#torch.zeros) (tensor filled with `0`s), [ones](https://pytorch.org/docs/1.6.0/generated/torch.ones.html?highlight=ones#torch.ones) (tensor filled with `1`s), [full](https://pytorch.org/docs/1.6.0/generated/torch.full.html?highlight=full#torch.full) (tensor filled with the specified value)
* [randn](https://pytorch.org/docs/1.6.0/generated/torch.randn.html?highlight=randn#torch.randn)
  (random tensor with values drawn from the normal distribution with mean `0` and variance `1`)

For each of those, there is a version (suffixed with `_like`) which takes on input a tensor and copies its shape instead (e.g. [full_like](https://pytorch.org/docs/1.6.0/generated/torch.full_like.html?highlight=full#torch.full_like)).

More specific tensor-creating functions also exist, for instance [eye](https://pytorch.org/docs/1.6.0/generated/torch.eye.html?highlight=eye#torch.eye) which allows to create 2-D (identity) tensors only. 


### Shape

Each tensor has a *shape*, which specifies its dimensions.  You can use the
`shape` attribute to access it.
```python
value_13.shape          # => torch.Size([])
value_13.shape          # => torch.Size([])
vector_1_to_10.shape    # => torch.Size([9])
matrix_1_to_10.shape    # => torch.Size([3, 3])
```

You can treat `torch.Size` objects as regular lists:
```python
len(value_13.shape)           # => 0
list(matrix_1_to_10.shape)    # => [3, 3]
matrix_1_to_10.shape[0]       # => 3
```

You cannot create tensors with irregular shapes.  This is not allowed:
```python
irregular_tensor = torch.tensor(
    [[1, 2, 3], 
     [4, 5, 6],
     [7, 8, 9, 10]])
# => ValueError: expected sequence of length 3 at dim 1 (got 4)
```

### dtype

The `dtype` attribute can be used to explicitely specify the type of values
stored in the tensor.
```python
# You can use the `dtype` attribute to enforce that the values be integers
int_vect = torch.tensor([1, 2, 2.5], dtype=torch.int64)
int_vect                  # => tensor([1, 2, 2])

# ... or floats
ints_as_floats_vect = torch.tensor([1, 2, 3], dtype=torch.float)
ints_as_floats_vect       # => tensor([1., 2., 3.])
```

<!---
TODO: bools, the `a = torch.randn(4, 2) < 0` syntax.
-->

### device

The target device (CPU, GPU) can be specified for each tensor separetely using
the `device` attribute.
```pyhon
# To run on CPU 
torch.tensor(0, device=torch.device("cpu"))

# To run on GPU (this will probably throw an exception on lab computers,
# where PyTorch was probably not compiled with CUDA enabled)
torch.tensor(0, device=torch.device("cuda:0"))
```
We will be using the CPU backend throughout the course.  PyTorch defaults to
CPU, so you don't really have to specify the device explicitly each time you
create a tensor (it won't hurt, though).

<!--
### Randomness

The `torch` module provides its own set of functions which allow to create
random tensors.  The main one is
[randn](https://pytorch.org/docs/stable/torch.html?highlight=randn#torch.randn),
which creates a tensor of the given shape with values drawn from the normal
distirubion (with mean `0` and variance `1`).
```python
# To get reproducible runs, set the randomness seed manually
torch.manual_seed(0)

# To create a 3x3 matrix, provide the shape as subsequent positional arguments
torch.randn(3, 3)
# tensor([[ 1.5410, -0.2934, -2.1788],
#         [ 0.5684, -1.0845, -1.3986],
#         [ 0.4033,  0.8380, -0.7193]])

# You can also provide a list
torch.manual_seed(0)
torch.randn([3, 3])
# tensor([[ 1.5410, -0.2934, -2.1788],
#         [ 0.5684, -1.0845, -1.3986],
#         [ 0.4033,  0.8380, -0.7193]])
```

*Note*: The way you initialize the parameters is actually pretty important.
Different initialization strategies work best for different architectures, and
the best strategies have been often determined empirically (based on trial and
error).  This is not particularly elegant, but deep learning is a lot about
finding the right balance between accuracy and speed, not necessarily elegance.
-->


<!--
### requires\_grad

In your typical PyTorch model, some of the tensors represent the parameters of
the model, others are the result of intermediate calculations.  For those that
represent parameters, use `requires_grad=True` when you create them.
```python
# A tensor parameter vector with five 0 floats
param_vect = torch.tensor([0.0 for _ in range(5)], requires_grad=True)
param_vect                # => tensor([0., 0., 0., 0., 0.], requires_grad=True)

# But typically you will want to randomly initialize your parameter tensor
param_vect = torch.randn([5], requires_grad=True)
param_vect		  # => tensor([-0.4062, -0.0975,  1.2838, -1.4413,  0.5867], requires_grad=True)
```

The reasons for using `requires_grad=True` are shortly explained below, in the
[Backpropagation](#backpropagation) section.
-->



## Operations

<!--
Below you can find a list of basic tensor operations should be sufficient to
build a large variety of different architectures (for example, a feed-forward
network).
-->

### Indexing and slicing

Indexing and slicing works similarly as with [numpy
arrays](https://www.pythoninformer.com/python-libraries/numpy/index-and-slice).
For instance:
```python
# Extract the first element of our vector
vector_1_to_10[0]         # => tensor(1)

# Slicing also works
vector_1_to_10[:3]        # => tensor([1, 2, 3])

# Extract the 3rd element of the 3rd row (two ways, second more efficient)
matrix_1_to_10[2][2]      # => tensor(9)
matrix_1_to_10[2, 2]      # => tensor(9)

# Slicing can be also used on several dimensions;
# To extract the 3rd row:
matrix_1_to_10[2, :]      # => tensor([7, 8, 9])
# To extract the 3rd column:
matrix_1_to_10[:, 2]      # => tensor([3, 6, 9])
# To remove the 1st row and the 1st column:
matrix_1_to_10[1:, 1:]    # => tensor([[5, 6],
                          #            [8, 9]])
```

Tensors are also
[iterable](https://docs.python.org/3.8/glossary.html#term-iterable):
```python
# Print all the elements in the vector
for x in vector_1_to_10:
    print(x)
# tensor(1)
# tensor(2)
# ...
# tensor(9)

# Print all the rows in the matrix
for x in matrix_1_to_10:
    print(x)
# tensor([1, 2, 3])
# tensor([4, 5, 6])
# tensor([7, 8, 9])
```
Need to iterate over columns?  One solution is to use
[transposition](https://pytorch.org/docs/1.6.0/generated/torch.t.html?highlight=t#torch.t)
or, more generally,
[permutation](https://pytorch.org/docs/1.6.0/tensors.html?highlight=permute#torch.Tensor.permute):
```
# Print all the columns in the matrix
for x in matrix_1_to_10.permute(1, 0):
    print(x)
# => tensor([1, 4, 7])
# => tensor([2, 5, 8])
# => tensor([3, 6, 9])
```

### View and reshape

TODO

### Element-wise operations

TODO

### Broadcasting

TODO

### Products

TODO

### Miscellaneous

TODO
