function nonprec_der = nonprecious_der_wrapper(der_func)
%NONPRECIOUS_DER_WRAPPER
%   NONPRECIOUS_DER_WRAPPER is used to implement derivatives of
%   reshaping functions, in a way that uses less memory.
%
%   Given a function (e.g. @reshape) which modifies the size of a tensor,
%   but is non precious (i.e. it doesn't need the data for the backwards
%   pass), a proxy function is used instead which expects a typed size of 
%   tensor on the back pass, instead of the actual tensor.

% Copyright (C) 2017 Ryan Webster
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).



  nonprec_der = str2func([func2str(der_func) '_nonprec']) ;
%   nonprec_der
  info = functions(nonprec_der) ;
%   info
  if isempty(info.file)
    % non native function or non reshaping function
    % just return the function
    nonprec_der = der_func;
  end
end

% same function as autonn_der, but expecting a size
function dx = reshape_der_nonprec(x, varargin) %#ok<*DEFNU>
  dy = varargin{end} ;
  if isscalar(dy) && dy == 0  % special case, no derivative
    dx = zeros(x, 'like', x) ;
  else
    dx = reshape(dy, x) ;
  end
end

% function dx = reshape_der_nonprec(x, varargin)  %#ok<*DEFNU>
%   dy = varargin{end} ;
%   if isscalar(dy) && dy == 0  % special case, no derivative
%     dx = zeros(size(x), 'like', x) ;
%   else
% %     size(dy)
% %     x
%     dx = reshape(dy, size(x)) ;
%   end
% end




