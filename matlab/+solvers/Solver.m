classdef Solver < handle
  %SOLVER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    learningRate = 0.001
    weightDecay = 0.0005
  end
  
  methods
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Solver arguments
      args = vl_parseprop(o, args, {'learningRate', 'weightDecay'}) ;
    end
    
    function step(o, net)
      % ensure supported training methods are ordered as expected
      assert(isequal(Param.trainMethods, {'gradient', 'average', 'none'})) ;

      % get parameter values and derivatives
      params = net.params ;
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
      net.setValue(idx, cell(size(idx))) ;
      
      
      % update gradient-based parameters, by calling subclassed solver
      is_grad = ([params.trainMethod] == 1) ;
      w(is_grad) = o.gradientStep(w(is_grad), dw(is_grad), lr(is_grad), decay(is_grad)) ;
      
      
      % update moving average parameters (e.g. batch normalization moments)
      is_avg = ([params.trainMethod] == 2) ;
      for i = find(is_avg)
        w{i} = vl_taccum(1 - lr(i), w{i}, lr(i) / params(i).fanout, dw{i}) ;
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
  end
  
end

