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
  opts.pad = 0 ;
  opts.batchNorm = false ;
  opts.batchNormScaleBias = true ;
  opts.batchNormEpsilon = 1e-4 ;
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
  
  % handle 'same' padding (i.e., maintain same output size, given stride 1)
  if isequal(opts.pad, 'same')
    pad = (opts.size(1:2) - 1) / 2 ;  % will be fractional for even filter sizes
    opts.pad = [floor(pad(1)), ceil(pad(1)), floor(pad(2)), ceil(pad(2))] ;
  end
  if opts.pad ~= 0  % don't include padding argument if it's 0
    convArgs(end+1:end+2) = {'pad', opts.pad} ;
  end
  
  % create conv layer
  out = vl_nnconv(in, 'size', opts.size, convArgs{:}) ;
  
  % prepare batch-norm arguments list
  if opts.batchNorm
    bnormArgs = {} ;
    if ~opts.batchNormScaleBias  % fixed scale and bias (constants instead of Params)
      bnormArgs(end+1:end+2) = {1, 0} ;
    end
    if opts.batchNormEpsilon ~= 1e-4  % non-default epsilon
      bnormArgs(end+1:end+2) = {'epsilon', opts.batchNormEpsilon} ;
    end
  end
  
  % create pre-activation batch norm
  if opts.batchNorm && opts.preActivationBatchNorm
    out = vl_nnbnorm(out, bnormArgs{:}) ;
  end
  
  % create activation layer
  out = createActivation(out, opts);
  
  % create post-activation batch norm
  if opts.batchNorm && ~opts.preActivationBatchNorm
    out = vl_nnbnorm(out, bnormArgs{:}) ;
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
