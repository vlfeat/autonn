function output = MaxoutNIN(varargin)
%MAXOUTNIN Returns a Maxout Network-in-Network for CIFAR10
%   Chang and Chen, "Batch-normalized maxout network in network", arXiv
%   2015. arXiv:1511.02583

% Copyright (C) 2018 Samuel Albanie, Jia-Ren Chang, Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 10 ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  % build network
  images = opts.input ;
  
  % first NIN block
  units = [192 160 96] ;  % maxout units per conv layer
  pieces = [1 5 5] ;  % maxout pieces per conv layer
  ker = [5 5] ;  % conv kernel
  poolKer = [3 3] ;  % pooling kernel
  pad = 2 ;  % input padding
  m1 = ninMaxoutBlock(images, 3, units, pieces, ker, pad, poolKer, false) ;
  outChannels = units(3) ;  % output channels of the NIN block
  
  % second NIN block
  units = [192 192 192] ;
  pieces = [1 5 5] ;
  ker = [5 5] ;
  poolKer = [3 3] ;
  pad = 2 ;
  m2 = ninMaxoutBlock(m1, outChannels, units, pieces, ker, pad, poolKer, false) ;
  outChannels = units(3) ;
  
  % third NIN block
  units = [192 192 opts.numClasses] ;
  pieces = [1 5 5] ;
  ker = [3 3] ;
  poolKer = [8 8] ;
  pad = 1 ;
  output = ninMaxoutBlock(m2, outChannels, units, pieces, ker, pad, poolKer, true) ;
  
  
  % default training options for this network
  defaults.numEpochs = 200 ;
  defaults.batchSize = 100 ;
  % the default learning rate schedule, changing every 50 epochs
  ep50 = ones(1, 50) ;
  defaults.learningRate = [0.5 * ep50, 0.05 * ep50, 0.005 * ep50, 0.0005 * ep50] ;
  output.meta = defaults ;
  
end

function block = ninMaxoutBlock(in, inChannels, ...
  units, pieces, ker, pad, poolKer, final)

  % first conv block
  sz = [ker(1:2), inChannels, units(1) * pieces(1)] ;
  c1 = vl_nnconv(in, 'size', sz, 'stride', 1, 'pad', pad) ;
  c1bn = vl_nnbnorm(c1) ;
  
  % second conv block
  sz = [1, 1, units(1) * pieces(1), units(2) * pieces(2)] ;
  c2 = vl_nnconv(c1bn, 'size', sz, 'stride', 1, 'pad', 0) ;
  c2bn = vl_nnbnorm(c2) ;
  
  % first maxout block
%   m1 = maxout(c2bn, units(2), pieces(2)) ;
  m1 = vl_nnmaxout(c2bn, pieces(2)) ;
  
  % third conv block
  sz = [1, 1, units(2), units(3) * pieces(3)] ;
  c3 = vl_nnconv(m1, 'size', sz, 'stride', 1, 'pad', 0) ;
  
  % skip if it's the final layer (prediction)
  if ~final
    c3bn = vl_nnbnorm(c3) ;

    % second maxout block
  %   m2 = maxout(c3bn, units(3), pieces(3)) ;
    m2 = vl_nnmaxout(c3bn, pieces(3)) ;

    % pooling and dropout
    p1 = vl_nnpool(m2, poolKer, 'method', 'avg', 'stride', 2, 'pad', [0 1 0 1]) ;

    block = vl_nndropout(p1, 'rate', 0.5) ;
  end
end

% function output = maxout(input, units, pieces)
%   % maxout is implemented by reshaping the tensor to split the channels
%   % dimension (assumed to have size = units*pieces) into 2 dimensions: the
%   % first has size = pieces, the second has size = units. the max operation
%   % is then ran along the pieces dimension, and the result is reshaped back
%   % to a 4D tensor.
%   
%   if pieces == 1  % trivial case, nothing to do
%     output = input ;
%     return
%   end
%   
%   rows = size(input, 1) ;
%   cols = size(input, 2) ;
%   
%   split = reshape(input, rows, cols, pieces, units, []) ;
%   
%   maxed = max(split, [], 3) ;
%   
%   output = reshape(maxed, rows, cols, units, []) ;
% end


