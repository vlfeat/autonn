function y = vl_nnmaxout(x, units, dzdy)
%VL_NNMAXOUT CNN maxout operator

  sz = size(x) ;
  sz(end + 1 : 4) = 1 ;  % ensure at least 4 dimensions
  
  assert(mod(sz(3), units) == 0, ...
    'The number of channels is not divisible by the number of maxout units.')
  
  blocks = sz(3) / units ;
  
  if nargin <= 2
    % forward pass
    
%     % merge spatial dim., and split channels into units and blocks dim.
%     split = reshape(x, prod(sz(1:2)), units, []) ;
%     maxed = max(split, [], 2) ;
    
    % merge spatial dimensions only
    split = reshape(x, prod(sz(1:2)), sz(3), []) ;
    maxed = vl_nnpool(split, [1 units], 'stride', [1 units]) ;

    y = reshape(maxed, [sz(1:2), blocks, sz(4:end)]) ;
  else
    % backward pass
    
    % merge spatial dimensions only
    split = reshape(x, prod(sz(1:2)), sz(3), []) ;

    split_der = reshape(dzdy, prod(sz(1:2)), blocks, []) ;
    
    maxed_der = vl_nnpool(split, [1 units], split_der, 'stride', [1 units]) ;

    y = reshape(maxed_der, sz) ;
  end
  
end

