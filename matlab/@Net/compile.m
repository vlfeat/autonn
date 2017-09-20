function compile(net, varargin)
%COMPILE Compile network
%   Main constructor for a Net, taking as input one or more Layers (the
%   network's outputs), and compiling them into a sequence of function
%   calls.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % parse options after the other inputs
  opts.sequentialNames = true ;
  opts.shortCircuit = true ;
  opts.optimizeGraph = true ;
  opts.forwardOnly = false ;  % used mainly by evalOutputSize for faster build
  [opts, netOutputs] = vl_argparsepos(opts, varargin) ;
  
  if ~isscalar(netOutputs)
    % several output layers; create a dummy layer to hold them together
    rootLayer = Layer(@root, netOutputs{:}) ;
  else
    rootLayer = netOutputs{1} ;
  end

  % make sure all layers have names
  if opts.sequentialNames
    rootLayer.sequentialNames() ;
  end

  % merge redundant inputs, this allow special inputs like
  % Input('testMode') to be used anywhere
  rootLayer.mergeRedundantInputs() ;
  
  % do graph optimizations, e.g. merging redundant weighted sums
  if opts.optimizeGraph
    rootLayer = rootLayer.optimizeGraph() ;
  end
  
  % list all layers again, now that they may have changed
  objs = rootLayer.find() ;
  
  % allocate an output variable to each one, sequentially
  for k = 1:numel(objs)
    objs{k}.outputVar = 2 * k - 1 ;
  end
  
  % do variable allocation optimizations, e.g. ReLU short-circuiting
  net.optimizeVars(opts, objs) ;
  

  % get meta properties from one Layer (ideally they should be merged)
  idx = find(cellfun(@(o) ~isempty(o.meta), objs)) ;
  if ~isempty(idx)
    assert(isscalar(idx), 'More than one Layer has the META property.') ;
    net.meta = objs{idx}.meta ;
  end

  % indexes of callable Layer objects (not Inputs, Params or Selectors)
  idx = find(cellfun(@(o) ~isa(o, 'Input') && ~isa(o, 'Param') && ~isa(o, 'Selector'), objs)) ;

  % allocate memory
  net.forward = Net.initStruct(numel(idx), 'func', 'name', ...
      'source', 'args', 'inputVars', 'inputArgPos', 'outputVar', 'outputArgPos', ...
      'debugStop','precious') ;
  net.backward = Net.initStruct(numel(idx), 'func', 'name', ...
      'source', 'args', 'inputVars', 'inputArgPos', 'numInputDer', 'accumDer') ;

  if opts.forwardOnly  % empty struct in this case, but with appropriate fields
    net.backward = net.backward([]);
  end
  
  % there is one var for the output of each Layer in objs; plus another
  % to hold its derivative.
  net.vars = cell(2 * numel(objs), 1) ;
  
  % whether a var is supposed to be on the GPU
  net.isGpuVar = false(size(net.vars)) ;

  numParams = nnz(cellfun(@(o) isa(o, 'Param'), objs)) ;
  net.params = Net.initStruct(numParams, 'name', 'var', ...
      'weightDecay', 'learningRate', 'source', 'trainMethod', 'fanout') ;
  net.inputs = struct() ;

  
  % first, handle Inputs, Params and Selectors
  p = 1 ;
  for i = 1:numel(objs)
    obj = objs{i} ;
    if isa(obj, 'Input')
      % an input, store its var index by name
      assert(~isempty(obj.name) && ~isfield(net.inputs, obj.name)) ;  % sanity check
      net.inputs.(obj.name) = obj.outputVar ;

      % set GPU state of corresponding var and derivative
      net.isGpuVar(obj.outputVar + [0, 1]) = obj.gpu ;
      
    elseif isa(obj, 'Param')
      % a learnable parameter, store them in a list
      net.params(p).var = obj.outputVar ;
      net.params(p).name = obj.name ;
      net.params(p).weightDecay = obj.weightDecay ;
      net.params(p).learningRate = obj.learningRate ;
      net.params(p).source = obj.source ;

      % store index of training method (defined in Param.trainMethods)
      net.params(p).trainMethod = find(strcmp(obj.trainMethod, Param.trainMethods)) ;
      
      % set GPU state of corresponding var and derivative
      net.isGpuVar(obj.outputVar + [0, 1]) = obj.gpu ;

      net.vars{obj.outputVar} = obj.value ;  % set initial value
      p = p + 1 ;
      
    elseif isa(obj, 'Selector')
      % handle layers with multiple outputs: each output selector attached
      % to a layer appends its own output var to that layer.
      obj.inputs{1}.outputVar(obj.index) = obj.outputVar ;
      
    elseif ~isempty(obj.numOutputs) && obj.numOutputs > numel(obj.outputVar)
      % layers with multiple outputs: assign a 0 index to any output vars
      % that exist even if they are unused (e.g. the above IF case does not
      % assign any index to them). this is so the backward build step is
      % aware of missing outputs, and assigns them a proper (0) derivative.
      obj.outputVar(end+1 : obj.numOutputs) = 0 ;
    end
  end
  
  
  % store functions for forward pass
  layer = [] ;
  for k = 1:numel(idx)
    obj = objs{idx(k)} ;
    layer.func = obj.func ;
    layer.name = obj.name ;
    layer.source = obj.source ;
    layer.outputArgPos = find(obj.outputVar ~= 0) ;  % skip unused outputs
    layer.outputVar = obj.outputVar(layer.outputArgPos) ;
    layer.debugStop = obj.debugStop ;
    layer.precious = obj.precious; 
    if numel(obj.numInputDer) && ~obj.numInputDer
        layer.precious = false;% non-differentiable functions are not precious
    end
    net.forward(k) = Net.parseArgs(layer, obj.inputs) ;
  end


  if ~opts.forwardOnly
    % store functions for backward pass
    layer = [] ;
    for k = numel(idx) : -1 : 1
      obj = objs{idx(k)} ;

      % add backward function to execution order
      layer.func = autonn_der(obj.func) ;
      layer.name = obj.name ;
      layer.source = obj.source ;
      layer.accumDer = obj.accumDer ;
      layer = Net.parseArgs(layer, obj.inputs) ;

      % figure out position of derivative argument: it's at the end of the
      % list, right before any property-value pairs or keywords, if they
      % exist (as determined by ISVARNAME).
      args = layer.args ;
      for lastInput = 0:numel(args)
        if lastInput < numel(args) && isvarname(args{lastInput + 1})
          break
        end
      end

      % figure out the number of returned values in bwd mode.
      % assume that the function in bwd mode returns derivatives for
      % all inputs until the last Layer input (e.g. if the 3rd input
      % has class Layer, and others are constants, assume there will be
      % at least 3 output derivatives).
      if isempty(obj.numInputDer)
        layer.numInputDer = max([0, layer.inputArgPos]) ;
      else  % manual override
        layer.numInputDer = obj.numInputDer ;
      end

      if layer.numInputDer == 0
        % there are no output derivatives, so this layer can be skipped
        layer.func = @deal ;
        [layer.args, layer.inputArgPos, layer.inputVars] = deal({}, [], []) ;

      else
        % store args for backward mode, with empty slots for der args
        numOutputDer = numel(obj.outputVar) ;
        layer.args = [args(1:lastInput), cell(1, numOutputDer), args(lastInput + 1 : end)] ;

        % modify argument positions according to the new empty slots
        next = layer.inputArgPos > lastInput ;
        layer.inputArgPos(next) = layer.inputArgPos(next) + numOutputDer ;

        % some outputs in forward mode may be unused. in this case, their
        % output derivatives will be 0. we need to treat them as constants.
        outputArgPos = (obj.outputVar ~= 0) ;
        layer.args(lastInput + find(~outputArgPos)) = {0} ;  % set them to 0
        
        % positions of der args
        layer.inputArgPos = [layer.inputArgPos, lastInput + find(outputArgPos)] ;

        % corresponding var indexes: the output derivatives of the layer
        % (recall that vars come in pairs, even-numbered are derivatives).
        outputVar = obj.outputVar(outputArgPos) ;  % skip the ignored outputs mentioned above
        layer.inputVars = [layer.inputVars, outputVar + 1] ;
      end
      layer = orderfields(layer);
      net.backward(numel(idx) - k + 1) = layer ;
    end
  end
  
  
  % network outputs, activate diagnostics automatically if empty
  for k = 1:numel(netOutputs)
    if isempty(netOutputs{k}.diagnostics)
      netOutputs{k}.diagnostics = true ;
    end
  end

  % compute varsFanOut, which is used to delete variables
  % not needed for the backwards pass (of non precious layers)
  net.varsFanOut = zeros(numel(net.vars),1);
  for k = 1:numel(net.forward)
      ii = net.forward(k).inputVars;
      if net.forward(k).precious
          net.varsFanOut(ii) = Inf; % an inelegant way to prevent deletion of vars in precious layers
      else
        net.varsFanOut(ii) =  net.varsFanOut(ii) + 1;
      end
  end
  
  % compute fan-out of parameters; this is useful to update batch-norm
  % moments with a moving average (cnn_train_autonn>accumulateGradientsAuto
  % NN). fan-out is the number of outgoing variables of a Param. take extra
  % care to *not* include edges that will not back-prop a derivative.
  % inputVars will concatenate indexes of all input vars, possibly repeated
  inputVars = cell(size(net.backward)) ;
  for k = 1:numel(net.backward)
    layer = net.backward(k) ;
    inputVars{k} = layer.inputVars(layer.inputArgPos <= layer.numInputDer) ;
  end
  inputVars = [inputVars{:}] ;
  % fan-out is just the number of occurences of the parameter in inputVars
  for k = 1:numel(net.params)
    net.params(k).fanout = nnz(inputVars == net.params(k).var) ;
  end
  
  
  % store diagnostics info for vars
  valid = false(numel(net.vars), 1) ;
  net.diagnostics = Net.initStruct(numel(net.vars), 'var', 'name') ;
  for k = 1 : numel(objs)
    if isequal(objs{k}.diagnostics, true)
      var = objs{k}.outputVar(1) ;
      net.diagnostics(var).var = var ;
      net.diagnostics(var).name = objs{k}.name ;
      net.diagnostics(var + 1).var = var + 1 ;
      net.diagnostics(var + 1).name = ['\partial ', objs{k}.name] ;
      valid([var, var + 1]) = true ;
    end
  end
  net.diagnostics(~valid) = [] ;
end
