# Building neural modules


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [nn.Module](#nnmodule)
- [Inheritance](#inheritance)
    - [Example: linear transformation](#example-linear-transformation)
    - [Example: FFN/MLP](#example-ffnmlp)
- [Composition](#composition)
    - [Example: FFN/MLP](#example-ffnmlp-1)
- [Parameters](#parameters)
- [Evaluation mode](#evaluation-mode)
- [Dropout](#dropout)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## nn.Module

A neural module<sup>1</sup> is basically a parameterised<sup>2</sup>,
differentiable<sup>3</sup> function which transforms input tensors to output
tensors.  In PyTorch, it is implemented by the [nn.Module][module] class and
its subclasses.  PyTorch modules are also building blocks, which can be
combined together to construct larger modules via [inheritance](#inheritance)
or [composition](#composition).

Links:
* ,,Deep Learning est mort. Vive Differentiable Programming''
* ,,Deep Learning is supervised learning of parameterised functions by gradient
  descent'' [link](https://www.signifytechnology.com/blog/2018/10/differentiable-functional-programming-by-noel-welsh)

**Note**: Using nn.Module is not obligatory (in the sense that most of what it
provides can be easily implemented manually), but it sure is very convenient
and much of the remaining PyTorch API relies on the nn.Module abstraction.

<!--
TODO: consider removing the "differentiable" word from the description above?
Maybe you want to mention that next time.
-->

<sup>1</sup> Not to be confused with a regular [Python
module](https://docs.python.org/3.8/tutorial/modules.html)!

<sup>2</sup> A neural module stores a list of its parameters, which can be
updated during training.

<sup>3</sup> A neural module is differentiable with respect to its parameters,
and PyTorch allows to use automatic differentiation to learn how these
parameters should be changed in order to better fit the entire neural model to
the data at hand.

## Inheritance

When creating a neural module via inheritance, you have to follow certain
rules:
* Use `nn.Module` (or its subclass) as the base class.
* Use `super().__init__()` at the beginning of the initialization method of
  **each class** that (directly or not) inherits from the PyTorch Module.
* Add sub-modules and parameters in the initialization method.  Simply assign them to the
  object's attributes.
* To use a [tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
  as a module's parameter, wrap it as a
  [Parameter](https://pytorch.org/docs/master/nn.html#torch.nn.Parameter) before
  assigning it to a module's attribute in the initialization method.
<!--
  Then you can treat it as a sub-module and assign to an attribute in the
  initialization method.
-->
* Finally, implement the function that the module represents in the `forward`
  method.

<!--
**Warning**: Remember that the sub-modules should not be used as the base class!
-->

#### Example: linear transformation

```python
import torch
import torch.nn as nn
from torch import Tensor

# Rule 1: use nn.Module as the base class
class Lin(nn.Module):

    """Linear transformation module.

    Type: `Tensor[N] -> Tensor[M]`, where:
    * `N` is the size of the input tensor
    * `M` is the size of the output tensor
    """

    def __init__(self, inp_size: int, out_size: int):
        # Rule 2: write `super().__init__()` at the beginning
        # of the initialization method
        super().__init__()
        # Rules 3 & 4: add sub-modules and parameters in the initialization
        # method & wrap the tensor parameters using the Parameter class
        self.M = nn.Parameter(torch.randn(out_size, inp_size))
        self.bias = nn.Parameter(torch.randn(out_size))

    # Rule 5: implement the function represented by the module using `forward`
    def forward(self, v: Tensor) -> Tensor:
        return torch.mv(self.M, v) + self.bias
```
You can then create and use a linear module as follows:
```python
# Create a module which transforms float tensors of size 3 to float tensors of
# size 5.
L = Lin(3, 5)

# Create a sample vector and apply the linear module to it
v = torch.randn(3)
L(v)
# TODO: ...
```

**Note**: this is just an example, PyTorch already provides an implementation
of a linear transformation: [nn.Linear][linear].

#### Parameters

A Module encapsulates the model parameters, which you can retrieve using the
[parameters](https://pytorch.org/docs/1.6.0/generated/torch.nn.Module.html?highlight=parameters#torch.nn.Module.parameters)
method.
```python
# Retrieve the parameters of the module
for param in L.parameters():
    print(param)
# TODO: Parameter containing:
# ...
# Parameter containing:
# ...
```

#### Exercises

**Exercise**: Use the official [nn.Linear][linear] PyTorch module to create a
linear transformation layer and apply it to vector `v`.  See how the parameters
of this module look like.

**Exercise**: Implement two-layered feed-forward network (FFN; also called
*multi-layered perceptron*, MLP) using inheritance.

**Exercise**: Factorize Linear as a combination of two modules and implement it
using composition.


## Composition

**Note**: The "composition" method for building modules is simpler and less
error-prone in practice than the [inheritance-based method](#inheritance).  Use
it when possible, use inheritance when you need extra flexibility.

In many cases, you just want to give the output from one module as input to
another module.
<!--(as in the [FFN example](#example_ffnmlp) above).
-->
In this case, the regular [function
composition](https://en.wikipedia.org/wiki/Function_composition) can be used to
combine the two modules.  To this end, PyTorch provides the
[Senuential][sequential] class.

**Note**: Whenever you use this method, make sure that the type and shape of
the output of the `i`-th module match with the type and shape of the input of
the `i+1`-th module.

#### Example: FFN

A two-layered feed-forward network, with ReLU activation, can be defined as
follows:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: list the rules
class FFN(nn.Module):
    def __init__(self, idim: int, hdim: int, odim: int):
        super(FFN, self).__init__()
        # Below, we create two `nn.Linear` sub-modules and assign them
        # to the attributes `lin1` and `lin2`.  They get automatically
        # registered as the FFN's sub-modules.
        self.lin1 = nn.Linear(idim, hdim)
        self.lin2 = nn.Linear(hdim, odim)

    def forward(self, x):
        # The following line is equivalent to: h = self.lin1.forward(x)
        y = self.lin1(x)
        # Apply ReLU and the second layer
        return self.lin2(F.relu(y))
```
You can then retrieve the module's parameters and apply it to a vector, as in
the linear transformation example (TODO: add link).
```python
# Create the FFN and retrieve its parameters
ffn = FFN(10, 5, 3)
ffn.parameters()        # Generator of parameters
list(ffn.parameters())  # To actually see the parameters

# Create an example vector of size 10 (input size) and push it through FFN
x = torch.randn(10)
y = ffn(x)
y.shape         # => torch.Size([3])
```

#### Exercises

**Exercise**: Implement FFN with a dynamic number of layers, whose shapes
should be specified as argument of the initialization method.

**Exercise**: Factorize Linear as a combination of two modules and implement it
using composition.


## Evaluation mode

<!--
Keeping track of all the parameters of the neural model is not the only
function of nn.Module.  Another, and very important one, is the ability to
switch the entire model between two modes: training (default) and evaluation.
-->

The forward calculation and the parameters are not the only things that a
PyTorch Module encapsulates.  Each module also keeps track of the current mode
(training vs evaluation) of the *entire model* (i.e., the main module + all the
sub-modules).

To retrieve the current mode:
```python
ffn.training    # True by default
assert ffn.training == ffn.lin1.training
                # All modules should be in the same mode
```

You can switch the mode using the `train` or `eval` method of the main module:
```python
ffn.eval()      # Set to evaluation mode
assert ffn.training == ffn.lin1.training
                # The mode of `ffn.lin1` should get updated, too
```

**WARNING**: You should never change the mode of the submodule, because this
will not propagate the mode information to other module components!
```python
ffn.eval()          # Set everything to evaluation mode
ffn.training        # => False
ffn.lin1.training   # => False
ffn.lin1.train()    # Set `ffn.lin1` to training mode
ffn.training        # => False
ffn.lin1.training   # => True (!!!)
```

<!--
## Dropout
-->



[module]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Module.html?highlight=module#torch.nn.Module "PyTorch neural module"
[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[sequential]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential "Sequential composition module"
