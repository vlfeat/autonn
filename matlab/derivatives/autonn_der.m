function derFunc = autonn_der(func)
%AUTONN_DER
%   AUTONN_DER is only called by Net during compilation.
%
%   Given a function handle, returns the function handle for its
%   derivative. It has the same name as the function, followed by '_der'.
%
%   Small derivative functions are defined as subfunctions here.

% Copyright (C) 2016-2017 Joao Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  derFunc = str2func([func2str(func) '_der']) ;
  info = functions(derFunc) ;
  
  % if '<func>_der' is undefined, '<func>' itself must implement the
  % derivative
  if isempty(info.file)
    derFunc = func ;
  end
end


function dx = reshape_der(x, varargin)  %#ok<*DEFNU>
  dy = varargin{end} ;
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(size(x), 'like', x) ;
  else
    dx = reshape(dy, size(x)) ;
  end
end

function dx = permute_der(x, dim, dy)
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(size(x), 'like', x) ;
  else
    dx = ipermute(dy, dim) ;
  end
end

function dx = ipermute_der(x, dim, dy)
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(size(x), 'like', x) ;
  else
    dx = permute(dy, dim) ;
  end
end

function dx = squeeze_der(x, dy)
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(size(x), 'like', x) ;
  else
    dx = reshape(dy, size(x)) ;
  end
end

function dx = flip_der(x, varargin)
  dy = varargin{end} ;
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(size(x), 'like', x) ;
  else
    dx = flip(dy, varargin{1:end-1}) ;  % undo flip
  end
end

function dx = rot90_der(x, k, dy)
  if nargin < 3
    dy = k ;  % derivative is second argument, not third
    k = 1 ;
  end
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(size(x), 'like', x) ;
  else
    dx = rot90(dy, -k) ;  % undo rotation
  end
end


function dx = abs_der(x, dy)
  assert(isreal(dy), 'Complex values not supported by ABS derivative.') ;
  dx = dy .* sign(x) ;
end

function dx = sqrt_der(x, dy)
  assert(all(x(:) > eps), 'Derivative undefined for SQRT(0) (approaches infinity), and for negative numbers.') ;
  dx = dy ./ sqrt(x) ;
end

function dx = exp_der(x, dy)
  dx = dy .* exp(x) ;
end

function dx = log_der(x, dy)
  assert(all(abs(x(:)) > eps), 'Derivative undefined for LOG(0) (approaches infinity).') ;
  dx = dy ./ x ;
end


function dx = sin_der(x, dy)
  dx = dy .* cos(x) ;
end

function dx = cos_der(x, dy)
  dx = -dy .* sin(x) ;
end

function dx = tan_der(x, dy)
  dx = dy .* sec(x).^2 ;
end

function dx = asin_der(x, dy)
  dx = dy ./ sqrt(1 - x.^2) ;
end

function dx = acos_der(x, dy)
  dx = -dy ./ sqrt(1 - x.^2) ;
end

function dx = atan_der(x, dy)
  dx = dy ./ (x.^2 + 1) ;
end

function [dy, dx] = atan2_der(y, x, da)
  r = max(x .* x + y .* y, eps);
  dx = -y ./ r .* da;
  dy = x ./ r .* da;
end


function dx = transpose_der(~, dy)
  dx = dy.' ;
end

function dx = ctranspose_der(~, dy)
  dx = dy' ;
end

function [da, db] = mtimes_der(a, b, dy)
  da = dy * b.' ;
  db = a.' * dy ;
end

function [da, db] = mrdivide_der(a, b, dy)
  % note: @mldivide is just @mrdivide with swapped inputs
  bt = b.' ;
  da = dy / bt ;
  db = -a' / bt * dy / bt ;
end

function dx = inv_der(x, dy)
  inv_x_t = inv(x)';
  dx = -inv_x_t * dy * inv_x_t;
end


function dx = sum_der(x, dim, dy)
  if nargin < 3
    % one-argument syntax of sum, plus derivative
    dy = dim;
    dim = find([size(x), 2] ~= 1, 1) ;  % find first non-singleton dim
  end

  % repeat dy along the summed dimension
  reps = ones(1, ndims(x)) ;
  reps(dim) = size(x,dim) ;
  dx = repmat(dy, reps) ;
end

function dx = mean_der(x, dim, dy)
  if nargin < 3
    % one-argument syntax of mean, plus derivative
    dy = dim;
    dim = find([size(x), 2] ~= 1, 1) ;  % find first non-singleton dim
  end

  % repeat dy along the summed dimension
  reps = ones(1, ndims(x)) ;
  reps(dim) = size(x,dim) ;
  dx = repmat(dy, reps) / size(x, dim) ;
end


function dx = gather_der(x, dy)
  if isa(x, 'gpuArray')
    dx = gpuArray(dy) ;  % convert derivative to same type as input
  else
    dx = dy ;  % keep same type (non-gpuArray)
  end
end



