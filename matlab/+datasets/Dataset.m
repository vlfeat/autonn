classdef Dataset < handle
  %DATASET Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    batchSize = 128  % batch size
    trainSet = []  % indexes of training samples
    valSet = []  % indexes of validation samples
    prefetch = false
  end
  
  methods
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Dataset arguments
      args = vl_parseprop(o, args, {'batchSize', 'prefetch'}) ;
    end
    
    function batches = train(o)
      % shuffle training set, and partition it into batches
      assert(~isempty(o.trainSet), ['To use the default dataset.train() ' ...
        'method, the training set must be specified (dataset.trainSet).']) ;
      
      batches = o.partition(o.trainSet(randperm(end))) ;
    end
    
    function batches = val(o)
      % partition validation set into batches
      assert(~isempty(o.valSet), ['To use the default dataset.val() ' ...
        'method, the validation set must be specified (dataset.valSet).']) ;
      
      batches = o.partition(o.valSet) ;
    end
    
    function batches = partition(o, idx, varargin)
      % partition indexes into batches (stored in a cell array).
      % if IDX is a matrix, each column is a distinct sample.
      if isvector(idx)
        idx = idx(:)' ;  % ensure row-vector
      end
      batches = cell(1, ceil(size(idx,2) / o.batchSize)) ;
      b = 1 ;
      for start = 1 : o.batchSize : size(idx,2)
        batches{b} = idx(:, start : min(start + o.batchSize - 1, end)) ;
        b = b + 1 ;
      end
      
      if o.prefetch
        batches = [batches; batches(1,2:end), {[]}] ;
      end
    end
  end
  
end

