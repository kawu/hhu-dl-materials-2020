## Backpropagation

*Backpropagation* is an algorithm that allows to methodically compute the
gradients of a complex expression using the chain rule, while caching
intermediary results.

This page provides examples and exercises regarding the implementation of
custom, backpropagation-enabled functions in PyTorch.  Such functions are
called *autograd* functions in PyTorch.  You will rarely need to manually
implement such functions.  Nevertheless, there are situations where this is
necessary, for instance:
* You may want to use a primitive function not implemented in PyTorch yet
  (by *primitive* I mean a function that is not easily expressible as a
  composition of already available PyTorch functions)
* Automatically derived backward calculation may be not optimal for certain
  combinations of neural functions

### Table of Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Useful links](#useful-links)
- [Examples](#examples)
  - [Preparation](#preparation)
  - [Addition](#addition)
  - [Product](#product)
  - [Composition](#composition)
- [Exercises](#exercises)
  - [Sum](#sum)
  - [Sigmoid](#sigmoid)
  - [Dot product](#dot-product)
  - [Matrix-vector product](#matrix-vector-product)
- [Footnotes](#footnotes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Useful links

The PyTorch documentation page which contains more detailed information about
writing custom autograd functions can be found at
https://pytorch.org/docs/master/notes/extending.html
<!---
https://pytorch.org/docs/stable/notes/extending.html.
-->

For a popular presentation of backpropagation and computation graphs, see
http://colah.github.io/posts/2015-08-Backprop.

<!---
Some code fragments were borrowed from:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
-->


## Examples

### Preparation

<!-- 
The commands and code fragments shown below are intended to be used
iteractivelly in the Python interpreter.  The code for this session can be also
found in the [backprop.py](backprop.py) file.
-->

We will use the following preamble:
```python
from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function
```

`Function` is the class we have to inherit from when we want to define custom
autograd functions in PyTorch.

<!---
However, you may want to perform the exercises below iteractivelly in IPython.
-->

### Addition

Let's start with a simple example: element-wise addition.  Of course it is
already implemented in PyTorch, which will allow us to test if our
implemenations work as intended.

```python
class Addition(Function):

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        z = x + y
        return z

    @staticmethod
    def backward(ctx, dldz: Tensor) -> Tuple[Tensor, Tensor]:
        return dldz, dldz
```
In the `forward` pass, we receive two tensors that we want to add together: `x`
and `y`.  To get the result of the forward method, we simply add them and
return the result.<sup>[1](#footnote1)</sup>

In the `backward` pass we receive a Tensor containing the gradient of the loss
`l` (whatever it is!) w.r.t the addition result `z`.  We call it `dldz`.  Now,
we need to calculate the gradients for `x` and `y` and return them as a
tuple, in the same order as in the `forward` method.  Using the chain rule, we
can determine that this is just `dldz` for both `x` and `y` (take a moment to
verify this!).

The addition function is now available via `Addition.apply`.  For brevity, it
is recommended to use an alias for custom autograd functions.  In this case:
```python
add = Addition.apply
```

We can now check that our custom addition behaves as the one provided in
PyTorch.
```python
x1 = torch.tensor(1.0, requires_grad=True)
y1 = torch.tensor(2.0, requires_grad=True)
(x1 + y1).backward()
```

We do the same with our custom addition function.
```python
x2 = torch.tensor(1.0, requires_grad=True)
y2 = torch.tensor(2.0, requires_grad=True)
add(x2, y2).backward()
```

And we verify that the gradients match.
```python
assert x1.grad == x2.grad
assert y1.grad == y2.grad
```

The nice part is that, since addition is element-wise, this should work also
for complex tensors, and not only for one-element
tensors!<sup>[2](#footnote2)</sup>  Let's see:
```python
x1 = torch.randn(3, 3, requires_grad=True)
y1 = torch.randn(3, 3, requires_grad=True)
(x1 + y1).sum().backward()

# We use `clone` and `detach` to get the exact, separate copies of x1 and y1.
x2 = x1.clone().detach().requires_grad_(True)
y2 = y1.clone().detach().requires_grad_(True)
add(x2, y2).sum().backward()

assert (x1.grad == x2.grad).all()
assert (y1.grad == y2.grad).all()
```

### Product

Let's take another example: element-wise product (multiplication).  Using the
chain rule, we can determine that:
* `dl/dx` = `dl/dz * y`
* `dl/dy` = `dl/dz * x`

In contrast with addition, to determine the partial derivatives `dl/dx` and
`dl/dy`, we need to have access to `y` and `x`, respectively, even though they
are not arguments of the `backward` method.

In the `forward` and `backward` methods, `ctx` is a context object that can be
used to stash information for the backward computation.  You can cache
arbitrary objects for use in the backward pass using the
`ctx.save_for_backward` method.  In our case, we can use it to stash `x` and
`y`:
```python
class Product(Function):

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        z = x * y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, dldz: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors
        return dldz*y, dldz*x

mul = Product.apply
```

Again, we can make sure the results are the same as with the regular,
element-wise operator `*`:
```python
# Using the regular element-wise multiplication operator
x1 = torch.tensor(3.0, requires_grad=True)
y1 = torch.tensor(2.0, requires_grad=True)
(x1 * y1).backward()

# Using the custom multiplication function
x2 = torch.tensor(3.0, requires_grad=True)
y2 = torch.tensor(2.0, requires_grad=True)
mul(x2, y2).backward()

# Verify that the computed gradients are the same
assert x1.grad == x2.grad
assert y1.grad == y2.grad
```
As with addition, this generalises to higher-dimensional tensors:
```python
x1 = torch.randn(3, 3, requires_grad=True)
y1 = torch.randn(3, 3, requires_grad=True)
(x1 * y1).sum().backward()

# We use `clone` and `detach` to get the exact, separate copies of x1 and y1.
x2 = x1.clone().detach().requires_grad_(True)
y2 = y1.clone().detach().requires_grad_(True)
mul(x2, y2).sum().backward()

assert (x1.grad == x2.grad).all()
assert (y1.grad == y2.grad).all()
```



<!--
### Sigmoid

Let's see another example: the sigmoid (logistic) function.

Let `x` be the input tensor, to which we apply the (element-wise) sigmoid
function.  Let `y` be the result of this application.  Finally, let `z` be the
loss value.

The derivative of sigmoid, `y = sigmoid(x) = 1 / (1 + exp(-x))`, is:
```
dy/dx = y * (1 - y)
```
From the chain rule we have:
```
dz/dx = dz/dy * dy/dx
```
Since in the backward computation we already know `dz/dy`, we need to also know
`y` (i.e., the result of the forward computation) to calculate `dy/dx` and,
subsequently, `dz/dx`.

In the `forward` and `backward` methods, `ctx` is a context object that can be
used to stash information for the backward computation.  You can cache
arbitrary objects for use in the backward pass using the
`ctx.save_for_backward` method.  In our case, we can use it to stash `y`:
```python
class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(y)
        return y
        
    @staticmethod
    def backward(ctx, dzdy: Tensor) -> Tensor:
        y, = ctx.saved_tensors
        return dzdy * y * (1 - y)

# Alias
sigmoid = Sigmoid.apply
```

To test it:
```python
x1 = torch.randn(3, 3, requires_grad=True)
z1 = torch.sigmoid(x1).sum()
z1.backward()

x2 = x1.clone().detach().requires_grad_(True)
z2 = sigmoid(x2).sum()
z2.backward()

# Check if the results of the forward computations are equal
assert (z1 == z2).all()

# Check if the two gradients are sufficiently similar (clearly the backward
# method of the PyTorch sigmoid is better in terms of numerical precision).
diff = x1.grad - x2.grad
assert (-1e-7 < diff).all()
assert (diff  < 1e-7).all()
```
-->

### Composition

As we can combine neural functions and modules, the underlying forward and
backward methods compose as well.

Let `a`, `b` and `c` be float tensors of the same shape (scalar tensors in the
simplest case), with `requires_grad=True`.<sup>[3](#footnote3)</sup>
Then, if we perform:
```python
mul(c, add(a, b)).sum().backward()
```
the order of computations is as follows:
* Forward: `d = add(a, b)`
* Forward: `e = mul(c, d)`
* Forward: `f = e.sum()`
* At this point, `f.backward()` is used
* Backward: `df/de` is calculated using `backward` from `sum`
* Backward: `df/dc` and `df/dd` are calculated using `backward` from `mul`
* Backward: `df/da` and `df/db` are calculated using `backward` from `add`

All this generalizes to arbitrary tensor-based computations, via the
abstraction called a *computation graph*.  More information on this can be
found at http://colah.github.io/posts/2015-08-Backprop.
<!--
and, hopefully, in the script.
-->

<!---
TODO: computation graph?
-->


## Exercises

**Note**: For all the exercises, you can use the *forward variants* of the
functions/methods already provided by PyTorch.  For instance, in the `sum`
exercise below, you can use `torch.sum` in the `forward` method.  We focus here
on the implementations of the `backward` methods.

<!--
**Note**: To solve some of the exercises below, you may need primitive
functions from the PyTorch library that we didn't use yet.
-->

### Sum

Re-implement `torch.sum` as a custom autograd function.  Verify that the
backpropagation results are the same as with the `torch.sum` function.

**Hint**: To obtain a concise solution, *broadcasting* may turn out useful.

### Sigmoid

Re-implement [`sigmoid`](https://en.wikipedia.org/wiki/Sigmoid_function) as a
custom autograd function.  Verify that the backpropagation results are the same
as with the `torch.sigmoid` function.  Note that the derivative of sigmoid has
a rather [simple
form](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x).

### Dot product

Re-implement `torch.dot` as a custom autograd function.

**Hint**: You can either represent dot product as a composition of two more
primitive functions and implement their autograd variants instead.  However,
the dot product implemented as a primitive autograd function may be more
efficient.

### Matrix-vector product

Re-implement `torch.mv` as a custom autograd function.

**Hint**. Use the multivariate chaine rule to solve this exercise on paper
first (you should actually do the calculations on paper first for other
exercises, too; the matrix-vector product exercise is simply more difficult to
solve, especially without proper preparations).

## Footnotes

<a name="footnote1">1</a>: Note that the addition operation (`x + y`) is
already backpropagation-enabled.  However, both autograd methods (`forward` and
`backward`) work in a setting where the gradient calculation is disabled (cf.
`torch.no_grad()`).
<!--
You can verify that by checking that no gradients are calculated in those two
methods.
-->

<a name="footnote2">2</a>: To show this formally, use the multivariate chaine
rule, which leads to the same formula as in the scalar case due to the
element-wise nature of the operation.

<a name="footnote3">3</a>: Our autograd functions `add` and `mul` do not
currently handle input arguments with `requires_grad=False`.  Support for such
cases can be added using the `ctx.needs_input_grad` attribute, for instance:
```python
class Addition(Function):

    @staticmethod
    def forward(ctx, x1: Tensor, x2: Tensor) -> Tensor:
        y = x1 + x2
        return y

    @staticmethod
    def backward(ctx, dzdy: Tensor) -> Tuple[Tensor, Tensor]:
        r1, r2 = ctx.needs_input_grad
        return dzdy if r1 else None, dzdy if r2 else None
```
See [extending
torch.autograd](https://pytorch.org/docs/master/notes/extending.html#extending-torch-autograd)
for more information.
