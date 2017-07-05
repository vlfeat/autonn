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
    
    function batches = partition(o, idx, varargin)
      % partition indexes into batches (stored in a cell array).
      % if IDX is a matrix, each column is a distinct sample.
      opts.prefetch = false ;
      opts = vl_argparse(opts, varargin) ;
      
      if isvector(idx)
        idx = idx(:)' ;  % ensure row-vector
      end
      batches = cell(1, ceil(size(idx,2) / o.batchSize)) ;
      b = 1 ;
      for start = 1 : o.batchSize : size(idx,2)
        batches{b} = idx(:, start : min(start + o.batchSize - 1, end)) ;
        b = b + 1 ;
      end
      
      if opts.prefetch
        batches = [batches; batches(1,2:end), {[]}] ;
      end
    end
  end
  
end

