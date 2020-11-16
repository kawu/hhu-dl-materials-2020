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
* Use `super().__init__()` at the beginning of the initialization method of
  **each class** that (directly or not) inherits from the PyTorch Module.
* Always add submodules in the initialization method.  Simply assign them to
  the object's attributes.
* In case you want to use a [raw
  tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) as a
  module's parameter, wrap it in the
  [Parameter](https://pytorch.org/docs/master/nn.html#torch.nn.Parameter)
  object.  Then you can treat it as a sub-module and assign to an attribute in
  the initialization method.

#### Example: linear transformation

```python
```

**Note**: this is just an example, if you want to use a linear transformation
module in your network, use [nn.Linear][linear] instead.

#### Example: FFN/MLP

TODO

Also, in the example, show how to retrieve the parameters of the module.


## Composition

**Note**: The "composition" method for building modules is simpler and less
error-prone in practice than the [inheritance-based method](#inheritance).  Use
it when possible, use inheritance when you need extra flexibility.

In many cases, you just want to give the output from one module as input to
another module (as in the [FFN example](#example_ffnmlp) above).  In this case,
the regular [function
composition](https://en.wikipedia.org/wiki/Function_composition) can be used to
combine the two modules.  To this end, PyTorch provides the
[Senuential][sequential] class.

**Note**: Whenever you use this method, make sure that the type and shape of
the output of the first module match with the type and shape of the input of
the second module.

#### Example: FFN/MLP

```python
TODO: example
```


## Parameters

## Evaluation mode

## Dropout



[module]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Module.html?highlight=module#torch.nn.Module "PyTorch neural module"
[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
[sequential]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential "Sequential composition module"
