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

**WARNING**:  You need to combine existing tensors to create new a new one?
Do not use `torch.tensor`, try
[stacking](https://pytorch.org/docs/1.6.0/generated/torch.stack.html?highlight=stack#torch.stack)
or
[concatenation](https://pytorch.org/docs/1.6.0/generated/torch.cat.html?highlight=cat#torch.cat)
instead.
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

There's also method
[dim](https://pytorch.org/docs/1.6.0/tensors.html?highlight=dim#torch.Tensor.dim)
which gives the number of dimensions:
```python
len(matrix_1_to_10.shape)     # => 2
matrix_1_to_10.dim()          # => 2
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
```python
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

### Iteration

Tensors are
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
```python
# Print all the columns in the matrix
for x in matrix_1_to_10.permute(1, 0):
    print(x)
# => tensor([1, 4, 7])
# => tensor([2, 5, 8])
# => tensor([3, 6, 9])
```

### View and reshape

In reality, a tensor is stored in memory as a contiguous chunk (a simple
array), while its *shape* allows to interpret it as a potentially
multi-dimensional entity.  Thanks to this, it is possible to reshape a tensor
without changing its contents using
[view](https://pytorch.org/docs/1.6.0/tensors.html?highlight=reshape#torch.Tensor.view)
or
[reshape](https://pytorch.org/docs/1.6.0/tensors.html?highlight=reshape#torch.Tensor.reshape).
```python
# View the matrix as a vector with 9 elements
matrix_1_to_10.view(9)     # => tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

# View the vector as a 3x3 matrix
vector_1_to_10.view(3, 3)  # => tensor([[1, 2, 3],
                           #            [4, 5, 6],
                           #            [7, 8, 9]])
```
**NOTE**: `view` is more restrictive than `reshape` but it is also more
efficient.

### Concatenation

Sometimes you need to join several existing tensors to create a new one.  As
[mentioned above](#creation), do not use `torch.tensor` for this!  Use
[stack](https://pytorch.org/docs/1.6.0/generated/torch.stack.html?highlight=stack#torch.stack)
or
[cat](https://pytorch.org/docs/1.6.0/generated/torch.cat.html?highlight=cat#torch.cat)
instead.
```python
# Stack vectors on top of each other to create a matrix
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
v3 = torch.tensor([7, 8, 9])
torch.stack([v1, v2, v3])  # => tensor([[1, 2, 3],
                           #            [4, 5, 6],
                           #            [7, 8, 9]])

# Concatenate the vectors
torch.cat([v1, v2, v3])	   # => tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Concatenate two copies of the matrix along the 1st dimension
torch.cat([matrix_1_to_10, matrix_1_to_10], dim=0)
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9],
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# Concatenate two copies of the matrix along the 2nd dimension
torch.cat([matrix_1_to_10, matrix_1_to_10], dim=1)
# tensor([[1, 2, 3, 1, 2, 3],
#         [4, 5, 6, 4, 5, 6],
#         [7, 8, 9, 7, 8, 9]])
```


### Element-wise operations

You can use the basic arithmetic operations on tensors: `+`, `-`, `*`, `/`.
They all work *element-wise*.
```python
# For one element tensors, this is pretty natural
x = torch.tensor(13)
assert x + 2 == 15          # Note the automatic cast from ints to int tensors

# For vectors
v = torch.tensor([1, 2, 3])
w = torch.tensor([1, 2, 1])
v * w                       # => tensor(1, 4, 3)

# For matrices
m1 = torch.tensor(
    [[1, 0],
     [0, 1]], dtype=torch.float)
m2 = torch.tensor(
    [[2, 2],
     [2, 2]], dtype=torch.float)
print(m1 / m2)
# tensor([[0.5000, 0.0000],
#         [0.0000, 0.5000]])
```

**Note**: the arguments must be of the same `dtype`.

### Boolean operations

Element-wise Boolean operations (`==`, `<`, `>`, etc.) provide a convenient way
to construct Boolean tensors.
```python
v = torch.tensor([1, 2, 3])
w = torch.tensor([1, 3, 1])
v == w                      # => tensor([ True, False, False])
v <  w                      # => tensor([False,  True, False])
v >= w                      # => tensor([ True, False,  True])
```
If you want to check if all values (any value) in a tensor are (is) `True`, use
[all](https://pytorch.org/docs/1.6.0/tensors.html?highlight=all#torch.BoolTensor.all)
([any](https://pytorch.org/docs/1.6.0/tensors.html?highlight=any#torch.BoolTensor.any)).
```python
(v == v).all()              # => tensor(True)
(v != v).any()              # => tensor(False)
(v != w).any()              # => tensor(True)
```

### Broadcasting

Element-wise operations (in particular) allow you to use tensors of different
shape provided that they [can be expanded to have the same
shape](https://pytorch.org/docs/1.6.0/notes/broadcasting.html?highlight=broadcasting).
```python
# Add 1 to a vector
v123 = torch.tensor([1, 2, 3])
v123 + 1                    # torch.tensor([2, 3, 4])

# Add the vector to the matrix
matrix_1_to_10 + v123       # => tensor([[ 2,  4,  6],
                            #            [ 5,  7,  9],
                            #            [ 8, 10, 12]])
```

### Products

PyTorch provides functions for various product operations:
* `torch.mv`: [matrix/vector product](https://pytorch.org/docs/1.6.0/generated/torch.mv.html?highlight=mv#torch.mv)
* `torch.mm`: [matrix/matrix product](https://pytorch.org/docs/1.6.0/generated/torch.mm.html?highlight=mm#torch.mm)
* `torch.bmm`: [batch matrix/matrix product](https://pytorch.org/docs/1.6.0/generated/torch.bmm.html?highlight=bmm#torch.bmm)

It also provides a powerful, generic function
[einsum](https://pytorch.org/docs/1.6.0/generated/torch.einsum.html?highlight=einsum#torch.einsum)
which can be used to implement each of those and more.
```python
v123 = torch.tensor([1, 2, 3])
torch.mv(matrix_1_to_10, v123)
# tensor([14, 32, 50])

torch.einsum("ij,j->i", matrix_1_to_10, v123)
# tensor([14, 32, 50])
```


### Miscellaneous

PyTorch of course provides many other functions and methods working on tensors,
both basic (e.g.,
[sum](https://pytorch.org/docs/1.6.0/tensors.html?highlight=sum#torch.Tensor.sum),
[abs](https://pytorch.org/docs/1.6.0/generated/torch.abs.html?highlight=abs#torch.abs), [dot
product](https://pytorch.org/docs/1.6.0/tensors.html?highlight=dot#torch.Tensor.dot), etc.)
and advanced (e.g.,
[logsumexp](https://pytorch.org/docs/1.6.0/generated/torch.logsumexp.html?highlight=logsumexp)).
Just search through the [documentation](https://pytorch.org/docs/1.6.0)
whenever you fill like the function/method you need should already have been
implemented.
