function output = BasicCifarNet(varargin)
%BASICCIFARNET Returns a simple network for CIFAR10

  % parse options. unknown arguments will be passed to ConvBlock (e.g.
  % activation).
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;
  opts.batchNorm = true ;  % whether to use batch normalization
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % get conv block generator with the given options. default activation is
  % ReLU, with pre-activation batch normalization (can be overriden).
  conv = models.ConvBlock('batchNorm', opts.batchNorm, ...
    'preActivationBatchNorm', true, 'weightScale', 0.01, convBlockArgs{:}) ;
  
  % build network
  images = opts.input ;
  
  x = conv(images, 'size', [5, 5, 3, 32], 'pad', 2, 'weightScale', 0.01) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'max', 'pad', 1) ;
  
  x = conv(x, 'size', [5, 5, 32, 32], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = conv(x, 'size', [5, 5, 32, 64], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = conv(x, 'size', [4, 4, 64, 64], 'weightScale', 0.05) ;
  
  output = conv(x, 'size', [1, 1, 64, opts.numClasses], 'weightScale', 0.05, ...
    'batchNorm', false, 'activation', 'none') ;
  
  
  % default training options for this network
  defaults.batchSize = 128 ;
  if ~opts.batchNorm
    defaults.learningRate = 0.01 ;
    defaults.numEpochs = 100 ;
  else
    defaults.learningRate = 0.1 ;
    defaults.numEpochs = 40 ;
  end
  output.meta = defaults ;
  
end
