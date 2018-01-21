classdef Dataset < handle
  %DATASET Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    batchSize = 128  % batch size
    trainSet = []  % indexes of training samples
    valSet = []  % indexes of validation samples
    partialBatches = false  % whether to return a partial batch (if the batch size does not divide the dataset size)
  end
  
  methods
    function dataset = Dataset(varargin)
      % parse generic Dataset arguments on non-subclassed construction.
      % allows Dataset to be used as a stand-alone class.
      varargin = dataset.parseGenericArgs(varargin) ;
      assert(isempty(varargin), 'Unknown arguments.') ;
    end
    
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Dataset arguments
      args = vl_parseprop(o, args, {'batchSize'}) ;
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
    
    function batches = partition(o, idx, batchSz)
      % partition indexes into batches (stored in a cell array).
      % if IDX is a matrix, each column is a distinct sample.
      if nargin < 3  % allow overriding batch size
        batchSz = o.batchSize ;
      end
      if isvector(idx)
        idx = idx(:)' ;  % ensure row-vector
      end
      batchSz = min(batchSz, size(idx,2));  % guard against big batch size
      batches = cell(1, ceil(size(idx,2) / batchSz)) ;
      b = 1 ;
      for start = 1 : batchSz : size(idx,2)
        batches{b} = idx(:, start : min(start + batchSz - 1, end)) ;
        b = b + 1 ;
      end
      
      % delete last partial batch if needed
      if ~o.partialBatches && size(batches{end}, 2) < batchSz
        batches(end) = [] ;
      end
    end
  end
  
end

