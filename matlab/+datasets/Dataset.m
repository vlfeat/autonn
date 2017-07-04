classdef Dataset < handle
  %DATASET Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    batchSize = 128
  end
  
  methods
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Dataset arguments
      args = vl_parseprop(o, args, {'batchSize'}) ;
    end
    
    function batches = partition(o, idx)
      % partition indexes into batches (stored in a cell array)
      batches = cell(1, ceil(numel(idx) / o.batchSize)) ;
      b = 1 ;
      for start = 1 : o.batchSize : numel(idx)
        batches{b} = idx(start : min(start + o.batchSize - 1, end)) ;
        b = b + 1 ;
      end
    end
  end
  
end

