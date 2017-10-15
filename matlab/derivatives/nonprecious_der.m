function nonprec_der = nonprecious_der(der_func)
%NONPRECIOUS_DER
%   NONPRECIOUS_DER is used to implement derivatives of
%   reshaping functions, in a way that uses less memory.
%
%   Given a function (e.g. @reshape) which modifies the size of a tensor,
%   but is non precious (i.e. it doesn't need the data for the backwards
%   pass), a proxy function is used instead which expects a size of the
%   tensor on the back pass, instead of the actual tensor.

% Copyright (C) 2017 Ryan Webster
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).



  nonprec_der = str2func([func2str(der_func) '_nonprec']) ;
  info = functions(nonprec_der) ;

  if isempty(info.file)
    % non native function or non reshaping function
    % just return the function
    nonprec_der = der_func;
  end
end


function dx = reshape_der_nonprec(x_sz, varargin) %#ok<*DEFNU>
  dy = varargin{end} ;
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz,'like',dy); 
  else
    dx = reshape(dy, x_sz) ;
  end
end

function dx = circshift_der_nonprec(x_sz, K,dim,dy)
  if nargin < 4 && isscalar(K) % 2016b behavior for scalar K
    dy = dim;
    dim = find([x_sz, 2] ~= 1, 1) ;  % find first non-singleton dim
  elseif nargin < 4 
    dy = dim;
  end
  
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz, 'like', dy) ;
  else
  	if nargin < 4 
      dx = circshift(dy,-K);
    else
      dx = circshift(dy,-K,dim);
    end
  end
end

function dx = sum_der_nonprec(x_sz, dim, dy)
  if nargin < 3
    % one-argument syntax of sum, plus derivative
    dy = dim;
    dim = find([x_sz, 2] ~= 1, 1) ;  % find first non-singleton dim
  end

  % repeat dy along the summed dimension
  reps = ones(1, numel(x_sz)) ;
  reps(dim) = x_sz(dim) ;
  dx = repmat(dy, reps) ;
end

function dx = permute_der_nonprec(x_sz, dim, dy)
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz, 'like', dy) ;
  else
    dx = ipermute(dy, dim) ;
  end
end

function dx = ipermute_der_nonprec(x_sz, dim, dy)
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz, 'like', dy) ;
  else
    dx = permute(dy, dim) ;
  end
end

function dx = squeeze_der_nonprec(x_sz, dy)
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz, 'like', dy) ;
  else
    dx = reshape(dy, x_sz) ;
  end
end


function dx = mean_der_nonprec(x_sz, dim, dy)
  if nargin < 3
    % one-argument syntax of mean, plus derivative
    dy = dim;
    dim = find([x_sz, 2] ~= 1, 1) ;  % find first non-singleton dim
  end

  % repeat dy along the summed dimension
  reps = ones(1, numel(x_sz)) ;
  reps(dim) = x_sz(dim) ;
  dx = repmat(dy, reps) / x_sz(dim) ;
end

function dx = flip_der_nonprec(x_sz, varargin)
  dy = varargin{end} ;
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz, 'like',dy) ;
  else
    dx = flip(dy, varargin{1:end-1}) ;  % undo flip
  end
end

function dx = rot90_der_nonprec(x_sz, k, dy)
  if nargin < 3
    dy = k ;  % derivative is second argument, not third
    k = 1 ;
  end
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x_sz, 'like', dy) ;
  else
    dx = rot90(dy, -k) ;  % undo rotation
  end
end


function varargout = cat_der_nonprec(dim, varargin)
%CAT_DER
  dzdy = varargin{end} ;
  
  if isscalar(dzdy)
    % special case, a scalar derivative propagates to all non-empty inputs
    valid = ~cellfun('isempty', varargin(1:end-1)) ;
    valid = [false, valid] ;  % add slot for DIM derivative, which is 0
    varargout = cell(size(valid)) ;
    varargout(valid) = {dzdy} ;
    varargout{1} = 0 ;
    return
  end

  % create indexing structure
  idx = cell(1, ndims(dzdy)) ;
  idx(:) = {':'} ;
  
  start = 1 ;
  varargout = cell(1, numel(varargin)) ;
  
  for i = 1 : numel(varargin) - 1
    % get size of this input along DIM; 0 if it is empty in any dimension.
    sz = varargin{i}(dim) * all(varargin{i}~=0) ;
    
    % retrieve corresponding derivative, by slicing dzdy
    idx{dim} = start : start + sz - 1 ;
    varargout{i + 1} = dzdy(idx{:}) ;
    start = start + sz ;
  end
  varargout{1} = 0 ;  % DIM derivative

end


function varargout = vl_nnwsum_nonprec(varargin)
%see VL_NNWSUM.
  assert(numel(varargin) >= 2 && isequal(varargin{end-1}, 'weights'), ...
    'Must supply the ''weights'' property.') ;

  w = varargin{end} ;  % vector of scalar weights
  n = numel(varargin) - 2 ;
  
  % this is only over called during the backward pass
  if n == numel(w) + 1
      % backward function (the last argument is the derivative)
      dy = varargin{n} ;
      n = n - 1 ;
      
      varargout = cell(1, n) ;
      for k = 1:n
        dx = dy ;
        for t = 1:ndims(dy)  % sum derivatives along expanded dimensions (by bsxfun)
          if varargin{k}(t) == 1  % original was a singleton dimension
            dx = sum(dx, t) ;
          end
        end
        varargout{k} = w(k) * dx ;
      end
      
  else
    error('The number of weights does not match the number of inputs.') ;
  end
end






