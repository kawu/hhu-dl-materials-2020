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
take as argument the desired shape of the tensor:
* [zeros](https://pytorch.org/docs/1.6.0/generated/torch.zeros.html?highlight=zeros#torch.zeros)
* [ones](https://pytorch.org/docs/1.6.0/generated/torch.ones.html?highlight=ones#torch.ones)
* [full](https://pytorch.org/docs/1.6.0/generated/torch.full.html?highlight=full#torch.full) and [full_like](https://pytorch.org/docs/1.6.0/generated/torch.full_like.html?highlight=full#torch.full_like)
* [randn](https://pytorch.org/docs/1.6.0/generated/torch.randn.html?highlight=randn#torch.randn)


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

### dtype and device

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


### View

*NOTE*: This part concerns internal representation of tensors.  This will be
quite useful during future sessions, but you can skip it on first reading.

TODO


## Tensor Operations

Below you can find a list of basic tensor operations that are sufficient to
build a large variety of different architectures (for example, a feed-forward
network).

### Basic element-wise operations

You can use the basic arithmetic operations on tensors: `+`, `-`, `*`, `/`.
They all work element-wise, i.e, the arguments have to have exactly the same
shape.
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

Additionally, the arguments have to have the same `dtype`.  For instance, the
following is not allowed:
```python
# This will raise an exception, because the first tensor keeps integers, the
# second tensor keeps floats, and PyTorch won't let you add them together.
torch.tensor([0, 1]) + torch.tensor([0.0, 1.0])
```

<!---
### Dot Product
-->

### Sum

You can sum all the elements in the given tensor using
[torch.sum](https://pytorch.org/docs/stable/torch.html#torch.sum).
```python
id = torch.tensor(
    [[1, 0],
     [0, 1]])
torch.sum(id)             # => tensor(2)
```

### Power

The [torch.pow](https://pytorch.org/docs/stable/torch.html#torch.sum) serves to
raise all the values in the tensor to the given power.
```python
v = torch.tensor([1, 2, 3])
torch.pow(v, 2)           # => tensor([1, 4, 9])
```

### Sigmoid

PyTorch provides a variety of non-linear functions
([sigmoid](https://pytorch.org/docs/stable/torch.html#torch.sigmoid),
[tanh](https://pytorch.org/docs/stable/torch.html#torch.tanh)).  They all apply
element-wise.  For instance, if you apply sigmoid to a vector, you actually
apply it to each of its elements individually.
```python
v = torch.tensor([1, 2, 3], dtype=torch.float) 
torch.sigmoid(v)           # => tensor([0.7311, 0.8808, 0.9526])

# We can apply sigmoid element-wise explicitely (this way is slower, though)
assert all(
    torch.sigmoid(v) ==
    torch.tensor([torch.sigmoid(x) for x in v]))
```

### Matrix-vector product

The [torch.mv](https://pytorch.org/docs/stable/torch.html#torch.mv) function
serves to perform a matrix-vector product.
```python
# Identity matrix of shape [2, 2]
id = torch.tensor(
    [[1, 0],
     [0, 1]])

# Example vector of shape [2]
v = torch.tensor([2, 3])

# Perform the matrix-vector product
assert all (torch.mv(id, v) == v), "Identity matrix doesn't change the input vector"
```

### Access

To access elements of tensors, you can basically treat them as lists (of lists
(of lists (...))).
```python
# Extract the first element of our vector
vector_1_to_10[0]         # => tensor(1)

# Print all the elements in the vector
for x in vector_1_to_10:
    print(x)
# tensor(1)
# tensor(2)
# ...
# tensor(9)

# The slicing syntax also works
vector_1_to_10[:3]        # => tensor([1, 2, 3])

# You can do the same with the matrix (then think of it as a list of lists)
for row in matrix_1_to_10: 
    print(row) 
# tensor([1, 2, 3])
# tensor([4, 5, 6])
# tensor([7, 8, 9])

# Extract the 3rd element of the 3rd row
x = matrix_1_to_10[2][2]
x                         # => tensor(9)
```

Whenever you access some parts or elements of tensors, you still get tensors in
return.  This is important because PyTorch models take the form of computations
over tensors, and the result of these these computations must typically be a
tensor, too.  The fact that, say, `vector_1_to_10[:3]` is a tensor means that
you can easily use it as a part of your PyTorch computation.

You can extract the raw values from one element tensors if you want.
```python
# You can extract the value of a one element tensor using `item()`
x.item()                  # => 9, regular int
# It doesn't work for higher-dimentional tensors
matrix_1_to_10.item()     
# => ValueError: only one element tensors can be converted to Python scalars
```


## Backpropagation

You will learn about backpropagation later during one of the theoretical
sessions.  Maybe we will also implement it at some point ourselves.

<!---
TODO: requires\_grad
-->

For now, the important things to know are:
* PyTorch model can be seen as a function from parameters (tensors with
`requires_grad=True`) to a *target* value.
* Backpropagation allows to calculate the gradient of the function, i.e., the
directions in which the individual parameters should be modified in order to
make the *target* value larger.

Typically, you don't even have to care about backpropagation. All the tensor
operations (addition, matrix-vector product, sum, etc.) that PyTorch provides
are *backpropagable*, i.e., you can use them transparently in your code and
PyTorch will be able to calculate the gradients of the model parameters.  In
fact, the way the API of the PyTorch library was designed is precisely to make
sure that, however you decide to combine the individual functions it provides,
you can still calculate the gradient.

At least, that's the idea, because in practice you can easily run into various
problems we will probably experience ourselves later during the course.

<!---
TODO: corner cases.
-->

<!---
*Note*: backpropagation is a specific case of automatic differentiation.
-->

<!---
To give an example, let's say we want to
-->

Let's see a minimal example.
```python
import torch

# Alias to tensor type
TT = torch.TensorType

# Target function we want to minimize
def f(x: TT, y: TT) -> TT:
    return (x - y + 1.0) ** 2.0

# Create two one element tensors and test our function
x = torch.tensor(0., requires_grad=True)
y = torch.tensor(0., requires_grad=True)
assert f(x, y) == torch.tensor(1.)

# Now let's find the gradient of `f` with respect to `x` and `y`.  To do that,
# we first use the `backward` method on the target value.
z = f(x, y)
z.backward()

# Then we access the gradient attributes of `x` and `y`
x.grad    		  # => 2.0
y.grad    		  # => -2.0

# Now we can move in the direction opposite to the gradient, which should make
# the value of the function `f` smaller.
with torch.no_grad():
    x -= x.grad * 0.1     # 0.1 is called the "learning rate"
    y -= y.grad * 0.1
f(x, y)                   # => tensor(0.3600, grad_fn=...)

# Now you can repeat these steps in a loop, and you obtain gradient descent!
for  i in range(0, 10):
    # Zero-out the gradients
    x.grad.zero_()
    y.grad.zero_()
    # Calculate the value of the function
    z = f(x, y)
    print("{iter_num}: {x}, {y} => {z}".format(
        iter_num=i,
        x=round(x.item(), 5),
        y=round(y.item(), 5),
        z=round(z.item(), 5))
        )
    # Backward computation
    z.backward()
    # Update the gradients
    with torch.no_grad():
        x -= x.grad * 0.1
        y -= y.grad * 0.1

```
