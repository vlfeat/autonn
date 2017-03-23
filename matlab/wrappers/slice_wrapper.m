function y = slice_wrapper(x, varargin)
% Implements the forward indexing/slicing operator as a function.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  y = x(varargin{:}) ;
end

