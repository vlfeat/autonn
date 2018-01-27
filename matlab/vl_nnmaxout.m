function y = vl_nnmaxout(x, units, dzdy)
%VL_NNMAXOUT CNN maxout operator
%   Goodfellow et al., "Maxout networks", arxiv 2013. arXiv:1302.4389

% Copyright (C) 2018 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  sz = size(x) ;
  sz(end + 1 : 4) = 1 ;  % ensure at least 4 dimensions
  
  assert(mod(sz(3), units) == 0, ...
    'The number of channels is not divisible by the number of maxout units.')
  
  blocks = sz(3) / units ;
  
  if nargin <= 2
    % forward pass
    
%     % native max() forward computation.
%     % merge spatial dim., and split channels into units and blocks dim.
%     split = reshape(x, prod(sz(1:2)), units, []) ;
%     maxed = max(split, [], 2) ;
    
    % alternative computation, based on non-overlapping max pooling.
    % first, merge spatial dimensions
    split = reshape(x, prod(sz(1:2)), sz(3), []) ;
    
    % max pooling
    maxed = vl_nnpool(split, [1 units], 'stride', [1 units]) ;

    % split spatial dimensions again
    y = reshape(maxed, [sz(1:2), blocks, sz(4:end)]) ;
  else
    % backward pass
    
    % use backward pass of non-overlapping max pooling, fast.
    % first, merge spatial dimensions
    split = reshape(x, prod(sz(1:2)), sz(3), []) ;
    split_der = reshape(dzdy, prod(sz(1:2)), blocks, []) ;
    
    % max pooling derivative
    maxed_der = vl_nnpool(split, [1 units], split_der, 'stride', [1 units]) ;

    % split spatial dimensions again
    y = reshape(maxed_der, sz) ;
  end
  
end

