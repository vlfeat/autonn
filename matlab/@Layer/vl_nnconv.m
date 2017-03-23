function layer = vl_nnconv(varargin)
%VL_NNCONV
%   Setup a conv layer: if there is a 'size' argument,
%   automatically initializes randomized Params for the filters.
%
%   The 'weightScale' argument specifies the initialization scale (or
%   'xavier' for Xavier initialization, which is the default).
%
%   Also handles 'hasBias' (initialize biases), 'learningRate' and
%   'weightDecay' arguments for the Params.

% Copyright (C) 2016 Joao F. Henriques.
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
