
# autonn #

# Getting started #

# Debugging your network #

# Documentation and examples #

The documentation is available by typing `help autonn/matlab` into the Matlab console, which lists all functions and classes.

Since autonn is a thin wrapper around the core MatConvNet functions, the inputs and options described in the core MatConvNet help files also apply to autonn (e.g., `help vl_nnconv`). Any additional options provided by autonn (such as syntatic sugar for initializing parameters) are described in the corresponding overloaded method of Layer (e.g. `help Layer.vl_nnconv`).

The easiest way to learn more is probably to look inside the `examples` directory, which has heavily-commented samples. These can be grouped in two categories:

The *minimal* examples (in `examples/minimal`) are very short and self-contained. They are scripts so you can inspect and explore the resulting variables in the command window. The SGD optimization is a simple `for` loop, so if you prefer to have full control over learning this is the way to go.

The *full* examples (in `examples/cnn` and `examples/rnn`) demonstrate training using `cnn_train_autonn`, equivalent to the MatConvNet `cnn_train` function. This includes the standard options, such as checkpointing and different solvers.

The MNIST and Imagenet examples work exactly the same as the corresponding MatConvNet examples, except for the network definitions. There is also a text LSTM example (`examples/rnn/rnn_lstm_shakespeare.m`), and a CNN on toy data (`examples/cnn/cnn_toy_data_autonn.m`), which provides a good basis for training on custom datasets.

