function layer = vl_nnconvt(varargin)
%VL_NNCONVT
%   Setup a conv-transpose layer. Simply create a conv layer first, then
%   switch the function handle.

  layer = vl_nnconv(varargin) ;
  layer.func = @vl_nnconvt ;

end

