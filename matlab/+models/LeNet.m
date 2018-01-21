function output = LeNet(varargin)
%LENET Returns a simple LeNet-5 for digit classification

  % parse options
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  x = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 20, 50], 'weightScale', 0.01) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  
  x = vl_nnconv(x, 'size', [4, 4, 50, 500], 'weightScale', 0.01) ;
  x = vl_nnrelu(x) ;
  
  output = vl_nnconv(x, 'size', [1, 1, 500, 10], 'weightScale', 0.01, ...
    'batchNorm', false, 'activation', 'none') ;

  % default training options for this network
  defaults.numEpochs = 100 ;
  defaults.batchSize = 128 ;
  defaults.learningRate = 0.001 ;
  output.meta = defaults ;
end
