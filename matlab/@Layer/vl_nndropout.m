function layer = vl_nndropout(x, varargin)
%VL_NNDROPOUT_SETUP
%   Setup a dropout layer, by adding a mask generator layer as input and
%   wrapping it.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.rate = 0.5 ;
  opts = vl_argparse(opts, varargin) ;

  % create mask generator layer
  maskLayer = Layer(@vl_nnmask, x, opts.rate) ;
  
  % create dropout wrapper layer
  layer = Layer(@vl_nndropout_wrapper, x, maskLayer, Input('testMode')) ;
  
  % vl_nndropout_wrapper doesn't return a derivative for the mask
  layer.numInputDer = 1 ;

end

