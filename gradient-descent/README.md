# Training by gradient descent

Plan:
* How to define a simple model (Q: POS tagging task?  language prediction
  task?)
* Loss function
* GD

<!-- START doctoc -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- END doctoc -->


## nn.Module

#### Usage

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

Here is how a linear transformation module can be implemented manually:
```python
TODO
```

**Note**: this is just an example, if you want to use a linear module in your
network, just use [nn.Linear][linear].


## Loss


[linear]: https://pytorch.org/docs/1.6.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear "Linear nn.Module"
