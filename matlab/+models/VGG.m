function output = VGG(varargin)
%VGG Returns a VGG-S/M/F model for ImageNet

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options. unknown arguments will be passed to ConvBlock (e.g.
  % activation).
  opts.pretrained = false ;  % whether to fetch a pre-trained model
  opts.input = Input('name', 'images', 'gpu', true) ;  % default input layer
  opts.numClasses = 1000 ;  % number of predicted classes
  opts.variant = 'm' ;  % choose between variants s/m/f (slow/medium/fast)
  opts.batchNorm = true ;  % whether to use batch normalization
  opts.preActivationBatchNorm = true ;  % whether batch-norm comes before or after activations
  opts.normalization = [5 1 0.0001/5 0.75] ;  % for LRN layer (vl_nnnormalize)
  [opts, convBlockArgs] = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  
  % default training options for this network (returned as output.meta)
  switch lower(opts.variant)
  case 's'
    meta.batchSize = 128 ;
  case 'm'
    meta.batchSize = 196 ;
  case 'f'
    meta.batchSize = 256 ;
  otherwise
    error('Unknown variant.') ;
  end
  meta.imageSize = [224, 224, 3] ;
  meta.augmentation.crop = 227 / 256;
  meta.augmentation.location = true ;
  meta.augmentation.flip = true ;
  meta.augmentation.brightness = 0.1 ;
  meta.augmentation.aspect = [2/3, 3/2] ;
  meta.weightDecay = 0.0005 ;
  
  % the default learning rate schedule
  if ~opts.pretrained
    if ~opts.batchNorm
      meta.learningRate = logspace(-2, -4, 60) ;
    else
      meta.learningRate = logspace(-1, -4, 20) ;
    end
    meta.numEpochs = numel(meta.learningRate) ;
  else  % fine-tuning has lower LR
    meta.learningRate = 1e-5 ;
    meta.numEpochs = 20 ;
  end
  
  
  % return a pre-trained model
  if opts.pretrained
    if opts.batchNorm
      warning('The pre-trained model does not include batch-norm (set batchNorm to false).') ;
    end
    if opts.numClasses ~= 1000
      warning('Model options are ignored when loading a pre-trained model.') ;
    end
    output = models.pretrained(['imagenet-vgg-' opts.variant]) ;
    
    % return prediction layer (not softmax)
    assert(isequal(output.func, @vl_nnsoftmax)) ;
    output = output.inputs{1} ;
    
    % replace input layer with the given one
    input = output.find('Input', 1) ;
    output.replace(input, opts.input) ;
    
    output.meta = meta ;
    return
  end
  
  
  % get conv block generator with the given options. default activation is
  % ReLU, with pre-activation batch normalization (can be overriden).
  conv = models.ConvBlock('batchNorm', opts.batchNorm, ...
    'preActivationBatchNorm', opts.preActivationBatchNorm, convBlockArgs{:}) ;
  
  
  % build network
  images = opts.input ;
  
  % implement the 3 variants: S, M and F (from paper)
  switch lower(opts.variant)
  case 's'
    % first conv block
    x = conv(images, 'size', [7, 7, 3, 96], 'stride', 2) ;
    if ~opts.batchNorm
      x = vl_nnnormalize(x, opts.normalization) ;
    end
    x = vl_nnpool(x, 3, 'stride', 3, 'pad', [0 2 0 2]) ;

    % second conv block
    x = conv(x, 'size', [5, 5, 96, 256]) ;
    if ~opts.batchNorm
      x = vl_nnnormalize(x, opts.normalization) ;
    end
    x = vl_nnpool(x, 2, 'stride', 2, 'pad', [0 1 0 1]) ;

    % conv blocks 3-5
    x = conv(x, 'size', [3, 3, 256, 512], 'pad', 1) ;
    x = conv(x, 'size', [3, 3, 512, 512], 'pad', 1) ;
    x = conv(x, 'size', [3, 3, 512, 512], 'pad', 1) ;
    x = vl_nnpool(x, 3, 'stride', 3, 'pad', [0 1 0 1]) ;

    % first fully-connected block
    x = conv(x, 'size', [6, 6, 512, 4096]) ;
    
  case 'm'
    % first conv block
    x = conv(images, 'size', [7, 7, 3, 96], 'stride', 2) ;
    if ~opts.batchNorm
      x = vl_nnnormalize(x, opts.normalization) ;
    end
    x = vl_nnpool(x, 3, 'stride', 2) ;

    % second conv block
    x = conv(x, 'size', [5, 5, 96, 256], 'stride', 2, 'pad', 1) ;
    if ~opts.batchNorm
      x = vl_nnnormalize(x, opts.normalization) ;
    end
    x = vl_nnpool(x, 3, 'stride', 2, 'pad', [0 1 0 1]) ;

    % conv blocks 3-5
    x = conv(x, 'size', [3, 3, 256, 512], 'pad', 1) ;
    x = conv(x, 'size', [3, 3, 512, 512], 'pad', 1) ;
    x = conv(x, 'size', [3, 3, 512, 512], 'pad', 1) ;
    x = vl_nnpool(x, 3, 'stride', 2) ;

    % first fully-connected block
    x = conv(x, 'size', [6, 6, 512, 4096]) ;

  case 'f'
    % first conv block
    x = conv(images, 'size', [11, 11, 3, 64], 'stride', 4) ;
    if ~opts.batchNorm
      x = vl_nnnormalize(x, opts.normalization) ;
    end
    x = vl_nnpool(x, 3, 'stride', 2, 'pad', [0 1 0 1]) ;

    % second conv block
    x = conv(x, 'size', [5, 5, 64, 256], 'pad', 2) ;
    if ~opts.batchNorm
      x = vl_nnnormalize(x, opts.normalization) ;
    end
    x = vl_nnpool(x, 3, 'stride', 2) ;

    % conv blocks 3-5
    x = conv(x, 'size', [3, 3, 256, 256], 'pad', 1) ;
    x = conv(x, 'size', [3, 3, 256, 256], 'pad', 1) ;
    x = conv(x, 'size', [3, 3, 256, 256], 'pad', 1) ;
    x = vl_nnpool(x, 3, 'stride', 2) ;

    % first fully-connected block
    x = conv(x, 'size', [6, 6, 256, 4096]) ;
  end

  % finish first fully-connected block
  if ~opts.batchNorm
    x = vl_nndropout(x) ;
  end
  
  % second fully-connected block
  x = conv(x, 'size', [1, 1, 4096, 4096]) ;
  if ~opts.batchNorm
    x = vl_nndropout(x) ;
  end

  % prediction layer
  output = conv(x, 'size', [1, 1, 4096, opts.numClasses]) ;

  output.meta = meta ;
  
end
