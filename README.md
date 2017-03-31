
# AutoNN #
AutoNN builds on [MatConvNet](http://www.vlfeat.org/matconvnet/)'s low-level functions and Matlab's math operators, to create a modern deep learning API with native automatic differentiation. The guiding principles are:

- Concise syntax for fast research prototyping, mixing math and deep network blocks freely.
- No boilerplate code to create custom layers, implemented as Matlab functions operating on GPU arrays.
- Minimal execution kernel for back-propagation, with a focus on speed.


# Requirements #

* Matlab 2015b or more recent (the newer, the better).
* MatConvNet [version 24](http://www.vlfeat.org/matconvnet/) or [more recent](https://github.com/vlfeat/matconvnet).


# Getting started #


```Matlab
```


# Documentation #

Comprehensive documentation is available by typing `help autonn` into the Matlab console. This lists all the classes and methods, with short descriptions, and provides links to each file's documentation.


# Examples #

The easiest way to learn more is probably to look inside the `examples` directory, which has heavily-commented samples. These can be grouped in two categories:

- The *minimal* examples (in `examples/minimal`) are very short and self-contained. They are scripts so you can inspect and explore the resulting variables in the command window. The SGD optimization is a simple `for` loop, so if you prefer to have full control over learning this is the way to go.

- The *full* examples (in `examples/cnn` and `examples/rnn`) demonstrate training using `cnn_train_autonn`, equivalent to the MatConvNet `cnn_train` function. This includes the standard options, such as checkpointing and different solvers.

The MNIST and ImageNet examples work exactly the same as the corresponding MatConvNet examples, except for the network definitions. There is also a text LSTM example (`examples/rnn/rnn_lstm_shakespeare.m`), and a CNN on toy data (`examples/cnn/cnn_toy_data_autonn.m`), which provides a good starting point for training on custom datasets.

