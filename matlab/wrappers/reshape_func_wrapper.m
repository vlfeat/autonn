function der_wrapper = reshape_der_wrapper(der_func)
%RESHAPE_DER_WRAPPER
%   RESHAPE_DER_WRAPPER is used to implement reshaping functions in a way
%   that is non precious
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



  der_wrapper = str2func([func2str(func) '_wrapper']) ;
  info = functions(func_lm) ;
  
  if isempty(info.file)
    error('Unimplemented reshaping function');
  end
end


function reshape_der_wrapper(

end




