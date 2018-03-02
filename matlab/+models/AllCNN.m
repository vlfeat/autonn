function prediction = AllCNN(varargin)
%ALLCNN Returns a All-Convolutional Network (All-CNN-C) for CIFAR10
%   M = models.AllCNN() returns variant C of the model proposed in:
%
%     Springenberg et al., "Striving for simplicity: The all convolutional
%     net", ICLR Workshop 2015.
%
%   models.AllCNN(..., 'option', value, ...) accepts the following options:
%
%   `input`:: default input
%     Specifies an input (images) layer for the network. If unspecified, a
%     new one is created.
%
%   `numClasses`:: 10
%     Number of output classes.
%
%   `batchNorm`:: true
%     Whether to use batch normalization.
%
%   Any other options will be passed to models.ConvBlock(), and can be used
%   to change the activation function, weight initialization, etc.
%
%   Suggested SGD training options are also returned in the struct M.meta.

  % parse options. unknown arguments will be passed to ConvBlock (e.g.
  % activation).
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;
  opts.batchNorm = true ;  % whether to use batch normalization
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % get conv block generator with the given options. default activation is
  % ReLU, with batch normalization (can be overriden).
  % also set a kernel size that will be reused by most layers.
  conv = models.ConvBlock('batchNorm', opts.batchNorm, ...
    'kernel', 3, convBlockArgs{:}) ;
  
  % build network
  images = opts.input ;
  
  % first stage
  x = conv(images, 'channels', [3, 96]) ;
  x = conv(x, 'channels', [96, 96]) ;
  x = conv(x, 'channels', [96, 96], 'stride', 2) ;
  
  if ~opts.batchNorm
    x = vl_nndropout(x, 'rate', 0.5) ;
  end
  
  % second stage
  x = conv(x, 'channels', [96, 192]) ;
  x = conv(x, 'channels', [192, 192]) ;
  x = conv(x, 'channels', [192, 192], 'stride', 2) ;
  
  if ~opts.batchNorm
    x = vl_nndropout(x, 'rate', 0.5) ;
  end
  
  % third stage. final conv has no batch-norm or activation
  x = conv(x, 'channels', [192, 192]) ;
  x = conv(x, 'channels', [192, 192], 'kernel', 1) ;
  x = conv(x, 'channels', [192, opts.numClasses], ...
    'kernel', 1, 'batchNorm', false, 'activation', 'none') ;
  
  % average pool predictions
  prediction = mean(mean(x, 1), 2) ;
  
  
  % default training options for this network
  defaults.batchSize = 128 ;
  defaults.weightDecay = 0.0005 ;
  if ~opts.batchNorm
    defaults.learningRate = 0.01 ;
    defaults.numEpochs = 100 ;
  else
    defaults.learningRate = 0.1 ;
    defaults.numEpochs = 40 ;
  end
  prediction.meta = defaults ;
  
end
