function layer = vl_nnconvt(varargin)
%VL_NNCONVT Overload for CNN convolution transpose
%   Y = Layer.vl_nnconvt(X, F, B) computes the convolution-transpose of the
%   image X with the filter bank F and biases B. See help vl_nnconvt for
%   more details.
%
%   This method overloads MatConvNet's vl_nnconvt function for Layer
%   objects, so that instead of executing vl_nnconvt, a new Layer object is
%   returned. X, F and B can be other Layers, including Params, or
%   constants.
%
%   The overloaded method accepts the same options as Layer.vl_nnconv.

  % simply create a conv layer first, then switch the function handle

  layer = vl_nnconv(varargin) ;
  layer.func = @vl_nnconvt ;

end

