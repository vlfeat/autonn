classdef SGD < solvers.Solver
  %SGD Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    momentum = 0.9
    state = {}  % momentum tensors
  end
  
  methods
    function o = SGD(varargin)
      % parse generic Solver arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this solver
      vl_parseprop(o, varargin, {'momentum'}) ;
    end
    
    function w = gradientStep(o, w, dw, lr, decay)
      % use local variables for speed
      momentum = o.momentum ;  %#ok<*PROPLC>
      state = o.state ;
      
      % initialize momentum state to 0
      if isempty(state)
        state = cell(size(w)) ;
        state(:) = {0} ;
      end
      
      for i = 1:numel(w)
        % update momentum
        state{i} = vl_taccum(momentum, state{i}, -1, dw{i}) ;

        % update parameters, incorporating weight decay
        w{i} = vl_taccum(1 - decay(i), w{i}, lr(i), state{i}) ;
      end
      
      o.state = state ;
    end
    
    function s = saveobj(o)
      % serialize to struct (called by the built-in function SAVE)
      % transfer state to CPU first
      s = o.saveGeneric() ;  % call parent class
      s.momentum = o.momentum ;
      s.state = cellfun(@gather, o.state, 'UniformOutput', false) ;
    end
  end
  
  methods (Static)
    function o = loadobj(s)
      % deserialize from struct (called by the built-in function LOAD)
      o = solvers.SGD() ;
      o.momentum = s.momentum ;
      o.state = s.state ;
      o.loadGeneric(s) ;  % call parent class
    end
  end
end

