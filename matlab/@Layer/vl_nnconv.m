function layer = vl_nnconv(varargin)
%VL_NNCONV
%   Y = Layer.vl_nnconv(X, F, B) computes the convolution of the image X
%   with the filter bank F and biases B. See help vl_nnconv for more
%   details.
%
%   This method overloads MatConvNet's vl_nnconv function for Layer
%   objects, so that instead of executing vl_nnconv, a new Layer object is
%   returned. X, F and B can be other Layers, including Params, or
%   constants.
%
%   In addition to those defined by MatConvNet's vl_nnconv, the overloaded
%   VL_NNCONV(..., 'option', value, ...) accepts the following options:
%
%   `size`:: []
%     If non-empty, automatically sets F and B to new Param objects,
%     initialized with random filters and zero biases of the given size.
%     The filters are normally-distributed, with Xavier scaling by default.
%     When using this syntax, F and B must be omitted.
%
%   `hasBias`:: true
%     Allows disabling the creation of the bias Param in the syntax above.
%
%   `weightScale`:: 'xavier'
%     If set to a real number, scales the filters by that number instead of
%     using Xavier initialization.
%
%   `learningRate`:: 1
%     Factor used to adjust the created Params' learning rate. Can specify
%     separate learning rates for F and B with a 2-elements vector.
%
%   `weightDecay`:: 1
%     Factor used to adjust the created Params' weight decay. Can specify
%     separate weight decays for F and B with a 2-elements vector.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % parse options. other options such as 'stride' will be maintained in
  % convArgs.
  opts = struct('size', [], 'weightScale', 'xavier', 'hasBias', true, ...
    'learningRate', 1, 'weightDecay', 1) ;
  [opts, posArgs, convOpts] = vl_argparsepos(opts, varargin) ;
  
  if ~isempty(opts.size)
    % a size was specified, create Params
    assert(numel(posArgs) == 1, ...
      'Must specify only one input Layer when using the ''size'' option.') ;
    
    if opts.hasBias
      % create bias as the 3rd input
      posArgs{3} = Param('value', zeros(opts.size(4), 1, 'single'), ...
                     'learningRate', opts.learningRate(max(1,end)), ...
                     'weightDecay', opts.weightDecay(max(1,end))) ;
    else
      posArgs{3} = [] ;
    end

    if isequal(opts.weightScale, 'xavier')
      scale = sqrt(2 / prod(opts.size(1:3))) ;
    else
      scale = opts.weightScale ;
    end

    % create filters as the 2nd input
    posArgs{2} = Param('value', randn(opts.size, 'single') * scale, ...
                    'learningRate', opts.learningRate(1), ...
                    'weightDecay', opts.weightDecay(1)) ;
  else
    assert(numel(posArgs) == 3, ...
      'Must specify all 3 inputs, or the ''size'' option.') ;
  end
  
  % create layer
  layer = Layer(@vl_nnconv, posArgs{:}, convOpts{:}) ;
end
