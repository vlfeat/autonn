function outputs = pretrained(modelName, varargin)
%PRETRAINED Loads a pre-trained model, possibly downloading it
%   A list of models is available at:
%   http://www.vlfeat.org/matconvnet/pretrained/

% Copyright (C) 2018 Samuel Albanie, Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.modelsDir = [vl_rootnn() '/data/models'] ;
  opts.modelsUrl = 'http://www.vlfeat.org/matconvnet/models' ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
  
  assert(exist(opts.modelsDir, 'dir'), 'Models directory does not exist.') ;
  
  % download if it doesn't exist
  modelPath = [opts.modelsDir '/' modelName '.mat'] ;
  if ~exist(modelPath, 'file')
    url = [opts.modelsUrl '/' modelName '.mat'] ;
    disp(['Model not found; attempting to download from: ' url]) ;
    websave(modelPath, url) ;
  end
  
  % load and convert to AutoNN layer
  net = load(modelPath) ;
  outputs = Layer(net) ;

end
