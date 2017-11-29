classdef State < Layer
%STATE

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    initialValue
  end
  
  methods
    function obj = State(varargin)
      opts.name = [] ;
      opts.initialValue = [] ;
      opts = vl_argparse(opts, varargin, 'nonrecursive') ;
      obj.name = opts.name ;
      obj.initialValue = opts.initialValue ;
    end
    
    function write(obj, value)
      obj.inputs{end+1} = value ;
    end
    
    function displayCustom(obj, ~, ~)
      s.name = obj.name ;
      s.initialValue = obj.initialValue ;
      s.inputs = obj.inputs ;
      fprintf('State\n\n') ;
      disp(s) ;
    end
  end
end
