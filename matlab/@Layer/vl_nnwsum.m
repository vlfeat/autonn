function layer = vl_nnwsum(varargin)
%VL_NNWSUM Overload for differentiable weighted sum
%   Y = Layer.vl_nnwsum(A, B, ..., 'weights', W) returns a weighted sum of
%   inputs, i.e. Y = W(1) * A + W(2) * B + ...
%   See help vl_nnwsum for more details.
%
%   This method overloads the vl_nnwsum function for Layer objects, so that
%   instead of executing vl_nnwsum, a new Layer object is returned. The
%   arguments A, B, ... can be Layer objects, or constants.

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % make sure there's a 'weights' property at the end, with the correct size
  assert(strcmp(varargin{end-1}, 'weights') && ...
    numel(varargin{end}) == numel(varargin) - 2) ;

  % separate inputs to the sum, and weights
  inputs = varargin(1:end-2) ;
  origWeights = varargin{end} ;
  weights = cell(size(inputs)) ;

  for k = 1 : numel(inputs)
    in = inputs{k} ;
    if isa(in, 'Layer') && isequal(in.func, @vl_nnwsum) && in.optimize
      % merge weights and store results
      inputs{k} = in.inputs(1:end-2) ;
      weights{k} = origWeights(k) * in.inputs{end} ;
    else
      % any other input (Layer or constant), wrap it in a single cell
      inputs{k} = {in} ;
      weights{k} = origWeights(k) ;
    end
  end

  % merge the results in order
  inputs = [inputs{:}] ;
  weights = [weights{:}] ;

  % append weights as a property
  inputs = [inputs, {'weights', weights}] ;
  
  % create layer
  layer = Layer(@vl_nnwsum, inputs{:}) ;
end

