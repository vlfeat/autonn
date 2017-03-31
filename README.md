
# AutoNN #
AutoNN builds on [MatConvNet](http://www.vlfeat.org/matconvnet/)'s low-level functions and Matlab's math operators, to create a modern deep learning API with native automatic differentiation. The guiding principles are:

- Concise syntax for fast research prototyping, mixing math and deep network blocks freely.
- No boilerplate code to create custom layers, implemented as Matlab functions operating on GPU arrays.
- Minimal execution kernel for back-propagation, with a focus on speed.

Compared to the previous [wrappers](http://www.vlfeat.org/matconvnet/wrappers/) for MatConvNet, AutoNN is less verbose and has lower computational overhead.


# Requirements #

* A recent Matlab (preferably 2015b onwards, though older versions may also work).
* MatConvNet [version 24](http://www.vlfeat.org/matconvnet/) or [more recent](https://github.com/vlfeat/matconvnet).


# Getting started #

Extract the AutoNN files somewhere, then pull up a Matlab console and get confortable! The first step is to add MatConvNet to the path (with `vl_setupnn`), as well as AutoNN (with `setup_autonn`).

## Defining networks ##

A deep neural network is a computational graph. With AutoNN, this graph is created by composing overloaded operators of special objects, much like in other modern frameworks that shall not be named. We start by defining one of the network's inputs:

```Matlab
images = Input();
```

We can then define the first operation, a convolution with a given size:

```Matlab
conv1 = vl_nnconv(images, 'size', [5, 5, 1, 20]);
```

The resulting object has class `Layer`. The `Input` we created earlier also subclasses `Layer`.

We can keep composing the result with other operations to build a whole network. For example, we can define a LeNet by composing convolutions, max-pooling and ReLUs:

```Matlab
pool1 = vl_nnpool(conv1, 2, 'stride', 2);
conv2 = vl_nnconv(pool1, 'size', [5, 5, 20, 50]);
pool2 = vl_nnpool(conv2, 2, 'stride', 2);
conv3 = vl_nnconv(pool2, 'size', [4, 4, 50, 500]);
relu1 = vl_nnrelu(conv3);
conv4 = vl_nnconv(relu1, 'size', [1, 1, 500, 10]);
```

This syntax creates parameters for filters and biases automatically, initialized with the Xavier method (see `help Layer.vl_nnconv` for other initialization options).

We can, of course, also create these parameters explicitly. They are objects of class `Param` (again, a subclass of `Layer`), and on creation we specify their initial value:

```Matlab
filters = Param('value', 0.01 * randn(5, 5, 1, 20, 'single'));
biases = Param('value', zeros(20, 1, 'single'));
```

Our `conv1` layer could then, alternatively, be defined as:

```Matlab
conv1 = vl_nnconv(images, filters, biases);
```

This follows the function signature for MatConvNet's `vl_nnconv` function exactly (which can be checked with `help vl_nnconv`). The difference is that, instead of calling `vl_nnconv` immediately, the function call's signature is stored in a `Layer` object, for later evaluation.

In general, you can build a layer based on a MatConvNet function by simply calling it with an argument of class `Layer`, since they are all overloaded.

Note also that the filters and biases for `vl_nnconv` don't have to be of class `Param`. They can be any other `Layer` (i.e., a subnetwork), or they can be simple Matlab arrays (and thus constant).






# Documentation #

Comprehensive documentation is available by typing `help autonn` into the Matlab console. This lists all the classes and methods, with short descriptions, and provides links to each file's documentation.


# Examples #

The easiest way to learn more is probably to look inside the `examples` directory, which has heavily-commented samples. These can be grouped in two categories:

- The *minimal* examples (in `examples/minimal`) are very short and self-contained. They are scripts so you can inspect and explore the resulting variables in the command window. The SGD optimization is a simple `for` loop, so if you prefer to have full control over learning this is the way to go.

- The *full* examples (in `examples/cnn` and `examples/rnn`) demonstrate training using `cnn_train_autonn`, equivalent to the MatConvNet `cnn_train` function. This includes the standard options, such as checkpointing and different solvers.

The MNIST and ImageNet examples work exactly the same as the corresponding MatConvNet examples, except for the network definitions. There is also a text LSTM example (`examples/rnn/rnn_lstm_shakespeare.m`), and a CNN on toy data (`examples/cnn/cnn_toy_data_autonn.m`), which provides a good starting point for training on custom datasets.

