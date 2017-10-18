function varargout = sort_der(x, varargin)
%SORT_DER
%   SORT_DER(X, DIM, ..., DZDY)
%   Derivative of SORT function. Same syntax as native SORT, plus derivative.
%  
% Copyright (C) 2017 Joao F. Henriques and Samuel Albanie.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  dzdy = varargin{end} ;

  if nargin == 2 % derivative of SORT(X)
    dim = 1 ; direction = 'ascend' ;
  elseif nargin == 3
    if ismember(varargin{2}, {'ascend', 'descend'}) 
      % derivative of SORT(X, DIRECTION)
      dim = 1 ; direction = varargin{2} ;
    else % derivative of SORT(X, DIM)
      dim = varargin{2} ; direction = 'ascend' ;
    end
  elseif nargin == 4 % derivative of SORT(X, DIM, DIERCTION)
    dim = varargin{2} ; direction = varargin{3} ;
  end 

  % The derivative of the sort function operates by computing the 
  % "reverse sort", mapping elements of DY to take the ordering of
  % X before it was sorted.
  [~,idx] = sort(x, dim, direction) ; % obtain original sort order
  outDims = cell(1, numel(size(x))) ;
  [outDims{:}] = ind2sub(size(x), 1:numel(x)) ;
  coords = vertcat(outDims{:})' ; % generate coordinates of inverse mapping
  revIdx = zeros(size(x)) ;
  for ii = 1:size(coords, 1)
    pos = num2cell(coords(ii,:)) ;
    pos{dim} = idx(pos{:}) ; % update relevant dimension of coord to undo sort
    revIdx(ii) = sub2ind(size(x), pos{:}) ;
  end
  inverted(revIdx) = 1:numel(x) ; % compute sort inverse indices
  y = dzdy{1}(inverted) ;
  varargout{1} = reshape(y, size(x)) ;
end

