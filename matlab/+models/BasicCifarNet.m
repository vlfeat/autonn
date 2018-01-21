function output = BasicCifarNet(varargin)
%BASICCIFARNET Returns a simple network for CIFAR10

  % parse options
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  x = vl_nnconv(images, 'size', [5, 5, 3, 32], 'pad', 2, 'weightScale', 0.01) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'max', 'pad', 1) ;
  x = vl_nnrelu(x) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 32, 32], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnrelu(x) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 32, 64], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnrelu(x) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = vl_nnconv(x, 'size', [4, 4, 64, 64], 'weightScale', 0.05) ;
  x = vl_nnrelu(x) ;
  
  output = vl_nnconv(x, 'size', [1, 1, 64, opts.numClasses], 'weightScale', 0.05, ...
    'batchNorm', false, 'activation', 'none') ;
  
  % default training options for this network
  defaults.numEpochs = 100 ;
  defaults.batchSize = 128 ;
  defaults.learningRate = 0.001 ;
  output.meta = defaults ;
  
end
