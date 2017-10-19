function varargout = sort_der(x, varargin)
%SORT_DER
%   SORT_DER(X, DIM, ..., DZDY)
%   Derivative of SORT function. Same syntax as native SORT, plus derivative.
%  
% Copyright (C) 2017 Samuel Albanie and Joao F. Henriques
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  dzdy = varargin{end} ;

  if nargin == 2 % derivative of SORT(X)
    dim = 1 ; direction = 'ascend' ;
  elseif nargin == 3
    if isnumeric(varargin{1}) % derivative of SORT(X, DIM)
      dim = varargin{1} ; direction = 'ascend' ;
    else % derivative of SORT(X, DIRECTION)
      dim = 1 ; direction = varargin{1} ;
    end
  elseif nargin == 4 % derivative of SORT(X, DIM, DIERCTION)
    dim = varargin{2} ; direction = varargin{3} ;
  end 

  % move onto CPU
  x_ = gather(x) ;

  % The derivative of the sort function operates by computing the 
  % "reverse sort", mapping elements of DY to take the ordering of
  % X before it was sorted.
  [~,idx] = sort(x_, dim, direction) ; % obtain original sort order
  outDims = cell(1, numel(size(x_))) ;
  [outDims{:}] = ind2sub(size(x_), 1:numel(x_)) ;
  coords = vertcat(outDims{:})' ; % generate coordinates of inverse mapping
  revIdx = zeros(size(x_)) ;
  sz = size(x_) ;
  %revIdx = arrayfun(@(ii) reverser(coords, dim, idx, ii, sz), 1:size(coords,1)) ;
  %keyboard
  coords2 = coords ;
  coords2(:,dim) = idx(:) ; % reverse along target dimension
  ins = mat2cell(coords2, size(coords2, 1), ones(1, size(coords2,2))) ;
  revIdx2 = sub2ind(sz, ins{:}) ;
  inverted(revIdx2) = 1:numel(x_) ; % compute sort inverse indices
  %revIdx2 = sub2ind(sz, coords2) ;
  %out = 1:size(coords,1) ;
  %newPos =  idx(1:numel(x_)) ; % dummy
  %keyboard
  %for ii = 1:size(coords, 1)
    %pos = num2cell(coords(ii,:)) ;
    %pos{dim} = idx(pos{:}) ; % update relevant dimension of coord to undo sort
    %revIdx(ii) = sub2ind(sz, pos{:}) ;
  %end
  %keyboard
  %inverted(revIdx) = 1:numel(x_) ; % compute sort inverse indices
  y = dzdy(inverted) ;
  varargout{1} = reshape(y, size(x_)) ;
end

function rev = reverser(coords, dim, idx, ii, sz)
    pos = num2cell(coords(ii,:)) ;
    pos{dim} = idx(pos{:}) ; % update relevant dimension of coord to undo sort
    rev = sub2ind(sz, pos{:}) ;
end
