
# AutoNN #
AutoNN builds on [MatConvNet](http://www.vlfeat.org/matconvnet/)'s low-level functions and Matlab's math operators, to create a modern deep learning API with native automatic differentiation. The guiding principles are:

- Concise syntax for fast research prototyping, mixing math and deep network blocks freely.
- No boilerplate code to create custom layers, implemented as Matlab functions operating on GPU arrays.
- Minimal execution kernel for back-propagation, with a focus on speed.

Compared to the previous [wrappers](http://www.vlfeat.org/matconvnet/wrappers/) for MatConvNet, AutoNN is less verbose and has lower computational overhead.


# Requirements #

* A recent Matlab (preferably 2015b onwards, though older versions may also work).
* MatConvNet [version 24](http://www.vlfeat.org/matconvnet/) or [more recent](https://github.com/vlfeat/matconvnet).


# AutoNN in a nutshell #

Add MatConvNet and AutoNN to the path (with the `vl_setupnn` and `setup_autonn` functions, respectively), then run the following:

```Matlab
% load simple data
s = load('fisheriris.mat') ;
data_x = single(s.meas.') ;  % features-by-samples matrix
[~, ~, data_y] = unique(s.species) ;  % convert strings to class labels

% define inputs and params
x = Input() ;
y = Input() ;
w = Param('value', 0.01 * randn(3, 4, 'single')) ;
b = Param('value', 0.01 * randn(3, 1, 'single')) ;

% combine them using math operators, which define the prediction
prediction = w * x + b ;

% compute least-squares loss
loss = sum(sum((prediction - y).^2)) ;

% assign names based on workspace variables, and compile net
Layer.workspaceNames() ;
net = Net(loss) ;

% simple SGD
learningRate = 1e-5 ;
outputs = zeros(1, 100) ;
rng(0) ;

for iter = 1:100,
  % draw minibatch
  idx = randperm(numel(data_y), 50) ;
  
  % evaluate network
  net.eval({x, data_x(:,idx), y, data_y(idx)'}) ;
  
  % update weights
  net.setValue(w, net.getValue(w) - learningRate * net.getDer(w)) ;
  net.setValue(b, net.getValue(b) - learningRate * net.getDer(b)) ;
  
  % plot loss
  outputs(iter) = net.getValue(loss) ;
end

figure(3) ; plot(outputs) ;
xlabel('Iteration') ; ylabel('Loss') ;
```


# Documentation #

## Tutorial ##

The easiest way to learn more is to follow this short [tutorial](https://github.com/vlfeat/autonn/blob/master/TUTORIAL.md). It covers all the basic concepts and a good portion of the API.


## Help pages ##

Comprehensive documentation is available by typing `help autonn` into the Matlab console. This lists all the classes and methods, with short descriptions, and provides links to other help pages.


## Examples ##

The `examples` directory has heavily-commented samples. These can be grouped in two categories:

- The *minimal* examples (in `examples/minimal`) are very short and self-contained. They are scripts so you can inspect and explore the resulting variables in the command window. The SGD optimization is a simple `for` loop, so if you prefer to have full control over learning this is the way to go.

- The *full* examples (in `examples/cnn` and `examples/rnn`) demonstrate training using `cnn_train_autonn`, equivalent to the MatConvNet `cnn_train` function. This includes the standard options, such as checkpointing and different solvers.

The ImageNet and MNIST examples work exactly the same as the corresponding MatConvNet examples, except for the network definitions. There is also a text LSTM example (`examples/rnn/rnn_lstm_shakespeare.m`), and a CNN on toy data (`examples/cnn/cnn_toy_data_autonn.m`), which provides a good starting point for training on custom datasets.

