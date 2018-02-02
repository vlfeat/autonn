classdef Adam < solvers.Solver
  %ADAM Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    beta1 = 0.9  % decay for first moment tensor
    beta2 = 0.999  % decay for second moment tensor
    eps = 1e-8  % additive offset to prevent division by zero
    
    m = {}  % first moment tensors
    v = {}  % second moment tensors
    t = 0  % iteration number, across epochs
  end
  
  methods
    function o = Adam(varargin)
      % parse generic Solver arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this solver
      vl_parseprop(o, varargin, {'beta1', 'beta2', 'eps'}) ;
    end
    
    function w = gradientStep(o, w, dw, lr, decay)
      % use local variables for speed
      [m, v, beta1, beta2, eps] = deal(o.m, o.v, o.beta1, o.beta2, o.eps) ;  %#ok<*PROPLC>
      
      % initialize all state variables to 0
      if isempty(m)
        m = cell(size(w)) ;
        m(:) = {0} ;
        v = m ;
        o.t = 0 ;
      end
      
      % update the time step
      o.t = o.t + 1 ;
      
      % this factor implicitly correct for biased estimates of first and
      % second moment vectors
      lr_factor = ((1 - beta2^o.t)^0.5) / (1 - beta1^o.t) ;
      
      for i = 1:numel(w)
        % incorporate weight decay into the gradient
        grad = vl_taccum(1, dw{i}, decay(i), w{i}) ;
        
        % update first moment vector, m
        m{i} = beta1 * m{i} + (1 - beta1) * grad ;

        % update second moment vector, v
        v{i} = beta2 * v{i} + (1 - beta2) * grad.^2 ;

        % update parameters
        w{i} = w{i} - lr(i) * lr_factor * m{i} ./ (v{i}.^0.5 + eps) ;
      end
      
      o.m = m ;
      o.v = v ;
    end
    
    function s = saveobj(o)
      % serialize to struct (called by the built-in function SAVE)
      % transfer state to CPU first
      s = o.saveGeneric() ;  % call parent class
      s.beta1 = o.beta1 ;
      s.beta2 = o.beta2 ;
      s.eps = o.eps ;
      s.m = cellfun(@gather, o.m, 'UniformOutput', false) ;
      s.v = cellfun(@gather, o.v, 'UniformOutput', false) ;
      s.t = o.t ;
    end
  end
  
  methods (Static)
    function o = loadobj(s)
      % deserialize from struct (called by the built-in function LOAD)
      o = solvers.Adam() ;
      o.beta1 = s.beta1 ;
      o.beta2 = s.beta2 ;
      o.eps = s.eps ;
      o.m = s.m ;
      o.v = s.v ;
      o.t = s.t ;
      o.loadGeneric(s) ;  % call parent class
    end
  end
end

