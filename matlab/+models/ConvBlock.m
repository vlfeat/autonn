function output = ConvBlock(varargin)
%CONVBLOCK Creates a conv block, with activation and optional batch-norm

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin > 0 && isa(varargin{1}, 'Layer')
    % create conv block immediately, with given layer as input
    output = createConvBlock(varargin{:}) ;
  else
    % return generator
    args = varargin ;  % generic arguments
    output = @(inputLayer, varargin) createConvBlock(inputLayer, args{:}, varargin{:}) ;
  end
end

function out = createConvBlock(in, varargin)
  % parse options
  opts.kernel = [3, 3] ;
  opts.inChannels = [] ;
  opts.outChannels = [] ;
  opts.size = [] ;
  opts.batchNorm = false ;
  opts.batchNormScaleBias = true ;
  opts.preActivationBatchNorm = false ;
  opts.activation = 'relu' ;
  opts.leak = 0.1 ;  % for leaky ReLU only
  [opts, convArgs] = vl_argparse(opts, varargin) ;
  
  if isempty(opts.size)
    % handle scalar kernel size (kernel is square)
    if isscalar(opts.kernel)
      opts.kernel = opts.kernel * [1, 1] ;
    end

    assert(~isempty(opts.inChannels), 'Must specify number of input channels.') ;
    assert(~isempty(opts.outChannels), 'Must specify number of output channels.') ;
    
    opts.size = [opts.kernel(1:2), opts.inChannels, opts.outChannels] ;
  end
  
  % create conv layer
  out = vl_nnconv(in, 'size', opts.size, convArgs{:}) ;
  
  % create pre-activation batch norm
  if opts.batchNorm && opts.preActivationBatchNorm
    if opts.batchNormScaleBias
      out = vl_nnbnorm(out) ;
    else
      out = vl_nnbnorm(out, 1, 0) ;  % fixed scale and bias
    end
  end
  
  % create activation layer
  out = createActivation(out, opts);
  
  % create post-activation batch norm
  if opts.batchNorm && ~opts.preActivationBatchNorm
    if opts.batchNormScaleBias
      out = vl_nnbnorm(out) ;
    else
      out = vl_nnbnorm(out, 1, 0) ;  % fixed scale and bias
    end
  end
end

function out = createActivation(in, opts)
  % create activation layer
  if isa(opts.activation, 'function_handle')
    out = opts.activation(in) ;  % custom function handle
    
  elseif isempty(opts.activation)
    out = in ;  % no activation
  else
    % standard activations
    switch lower(opts.activation)
    case 'relu'
      out = vl_nnrelu(in) ;
    case 'leakyrelu'
      out = vl_nnrelu(in, 'leak', opts.leak) ;
    case 'sigmoid'
      out = vl_nnsigmoid(in) ;
    case 'none'
      out = in ;
    otherwise
      error('Unknown activation type.') ;
    end
  end
end
