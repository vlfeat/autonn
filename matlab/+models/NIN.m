function output = NIN(varargin)
%NIN Returns a Network-in-Network model for CIFAR10
%   Lin, Chen and Yan, "Network in network", arXiv 2013. arXiv:1312.4400

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options. unknown arguments will be passed to ConvBlock (e.g.
  % batchNorm).
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;  % number of predicted classes
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  % first NIN block
  channels = [192 160 96] ;  % number of outputs channels per conv layer
  ker = [5 5] ;  % conv kernel
  poolKer = [3 3] ;  % pooling kernel
  poolMethod = 'max' ;  % pooling method
  pad = 2 ;  % input padding
  m1 = ninBlock(images, 3, channels, ker, pad, ...
    poolKer, poolMethod, convBlockArgs, false) ;
  outChannels = channels(3) ;  % output channels of the NIN block
  
  % second NIN block
  channels = [192 192 192] ;
  ker = [5 5] ;
  poolKer = [3 3] ;
  poolMethod = 'avg' ;
  pad = 2 ;
  m2 = ninBlock(m1, outChannels, channels, ker, pad, ...
    poolKer, poolMethod, convBlockArgs, false) ;
  outChannels = channels(3) ;
  
  % third NIN block
  channels = [192 192 opts.numClasses] ;
  ker = [3 3] ;
  poolKer = [7 7] ;
  poolMethod = 'avg' ;
  pad = 1 ;
  output = ninBlock(m2, outChannels, channels, ker, pad, ...
    poolKer, poolMethod, convBlockArgs, true) ;
  
  
  % default training options for this network
  defaults.batchSize = 100 ;
  % the default learning rate schedule
  defaults.learningRate = [0.002, 0.01, 0.02, 0.04 * ones(1,80), 0.004 * ones(1,10), 0.0004 * ones(1,10)] ;
  defaults.numEpochs = numel(defaults.learningRate) ;
  output.meta = defaults ;
  
end

function block = ninBlock(in, inChannels, outChannels, ...
  ker, pad, poolKer, poolMethod, convBlockArgs, final)

  % get conv block generator with the given options.
  % default activation is ReLU, with post-activation batch normalization.
  conv = models.ConvBlock('batchNorm', true, convBlockArgs{:}) ;
  
  % 2 conv blocks
  c1 = conv(in, 'size', [ker(1:2), inChannels, outChannels(1)], 'pad', pad) ;
  c2 = conv(c1, 'size', [1, 1, outChannels(1), outChannels(2)]) ;
  
  if ~final
    % third conv block
    c3 = conv(c2, 'size', [1, 1, outChannels(2), outChannels(3)]) ;
    
    % pooling and dropout
    p1 = vl_nnpool(c3, poolKer, 'method', poolMethod, 'stride', 2) ;
    block = vl_nndropout(p1, 'rate', 0.5) ;
  else
    % it's the final layer (prediction), no batch-norm/activation/pool
    block = conv(c2, 'size', [1, 1, outChannels(2), outChannels(3)], ...
      'batchNorm', false, 'activation', 'none') ;
  end
end

