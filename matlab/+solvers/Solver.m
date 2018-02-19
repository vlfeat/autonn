classdef Solver < handle
  %SOLVER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    learningRate = 0.001
    weightDecay = 0
    conserveMemory = true
  end
  
  methods
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Solver arguments
      args = vl_parseprop(o, args, {'learningRate', 'weightDecay'}) ;
    end
    
    function step(o, net, varargin)
      % takes one step of the solver, using the given network's gradients.
      % can specify a subset of parameters to affect exclusively, or to
      % ignore.
      opts.affectParams = [] ;
      opts.ignoreParams = [] ;
      opts = vl_argparse(opts, varargin, 'nonrecursive') ;
      
      % ensure supported training methods are ordered as expected
      assert(isequal(Param.trainMethods, {'gradient', 'average', 'none'})) ;
      
      params = net.params ;
      
      % select a set of parameters to affect, or ignore
      affected = [];
      negate = false;
      if ~isempty(opts.affectParams)
        affected = opts.affectParams ;
        assert(isempty(opts.ignoreParams), ...
          'Cannot specify parameters to ignore and affect simultaneously.') ;
        
      elseif ~isempty(opts.ignoreParams)
        affected = opts.ignoreParams ;
        negate = true ;
      end

      % match variable indexes to params, and keep only specified subset
      if ~isempty(affected)
        affectedVars = net.getVarIndex(affected) ;
        affectParams = ismember([params.var], affectedVars);
        if negate
          affectParams = ~affectParams ;
        end
        params = params(affectParams) ;
      end
      
      % get parameter values and derivatives
      idx = [params.var] ;
      w = net.getValue(idx) ;
      dw = net.getDer(idx) ;
      if isscalar(idx)
        w = {w} ; dw = {dw} ;
      end
      
      % final learning rate and weight decay per parameter
      lr = [params.learningRate] * o.learningRate ;
      decay = [params.weightDecay] * o.weightDecay ;
      
      % allow parameter memory to be released
      if o.conserveMemory
        net.setValue(idx, cell(size(idx))) ;
      end
      
      
      % update gradient-based parameters, by calling subclassed solver
      is_grad = ([params.trainMethod] == 1) ;
      w(is_grad) = o.gradientStep(w(is_grad), dw(is_grad), lr(is_grad), decay(is_grad)) ;
      
      
      % update moving average parameters (e.g. batch normalization moments)
      is_avg = ([params.trainMethod] == 2) ;
      lr_avg = [params.learningRate] ;  % independent learning rate
      for i = find(is_avg)
        w{i} = vl_taccum(1 - lr_avg(i), w{i}, lr_avg(i) / params(i).fanout, dw{i}) ;
      end
      
      
      % write values back to network
      if isscalar(idx)
        w = w{1} ;
      end
      net.setValue(idx, w) ;
    end
    
    function w = gradientStep(o, w, dw, learningRates, weightDecays)  %#ok<INUSD>
      error('Cannot instantiate Solver directly; use one of its subclasses (e.g. solvers.SGD).');
    end
    
    function s = saveGeneric(o)
      % must be called by subclass's saveobj (called by built-in SAVE)
      s.learningRate = o.learningRate ;
      s.weightDecay = o.weightDecay ;
      s.conserveMemory = o.conserveMemory ;
    end
    
    function loadGeneric(o, s)
      % must be called by subclass's loadobj (called by built-in SAVE)
      o.learningRate = s.learningRate ;
      o.weightDecay = s.weightDecay ;
      if isfield(s, 'conserveMemory') 
        o.conserveMemory = s.conserveMemory ; 
      else 
        o.conserveMemory = true ; 
      end
    end
  end
  
end

