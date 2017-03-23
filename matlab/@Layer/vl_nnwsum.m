function layer = vl_nnwsum(varargin)
%VL_NNWSUM_SETUP
%   Setup a weighted sum layer, by merging any other weighted sums in its
%   inputs.

% Copyright (C) 2016 Joao F. Henriques.
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

