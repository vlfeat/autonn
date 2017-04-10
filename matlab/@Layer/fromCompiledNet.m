function netOutputs = fromCompiledNet(net)
%FROMCOMPILEDNET Decompiles a Net back into Layer objects
%   OUTPUTS = Layer.fromCompiledNet(NET) decompiles a compiled network (of
%   class Net), into their original Layer objects (i.e., a set of
%   recursively nested Layer objects).
%
%   Returns a cell array of Layer objects, each corresponding to an output
%   of the network. These can be composed with other layers, or compiled
%   into a Net object for training/evaluation.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
  

  % use local copies so that any changes won't reflect on the original Net
  forward = net.forward ;
  backward = net.backward ;


  % first undo ReLU short-circuiting, since it creates cycles in the graph.
  % collect unused vars left behind by short-circuiting, to reuse them now.
  unused = true(size(net.vars)) ;
  unused([forward.inputVars]) = false ;
  unused([forward.outputVar]) = false ;
  unused(2:2:end) = false ;  % ignore derivative vars
  unused = find(unused) ;  % convert to list of var indexes
  
  next = 1 ;  % next unused var to take
  for k = 1:numel(forward)
    if isequal(forward(k).func, @vl_nnrelu) && ...
      isequal(forward(k).inputVars, forward(k).outputVar)

      % a short-circuited ReLU (output var = input var). change its output
      % to a new unused var, and update any layers that depend on it.
      oldVar = forward(k).outputVar ;
      for j = k + 1 : numel(forward)
        in = forward(j).inputVars ;
        if any(in == oldVar)  % update this dependent layer
          forward(j).inputVars(in == oldVar) = unused(next) ;
        end
      end
      
      forward(k).outputVar = unused(next) ;  % update the ReLU
      
      next = next + 1 ;
      assert(next <= numel(unused) + 1) ;  % shouldn't happen if short-circ. ReLUs leave unused vars behind
    end
  end
  

  % main decompilation code.
  
  % create reverse look-up for output vars in the execution order
  var2forward = zeros(size(forward));
  for k = 1:numel(forward)
    var2forward(forward(k).outputVar) = k ;
  end

  % create a cell array 'layers' that will contain all created Layers,
  % indexed by their output vars. e.g., if layer L outputs var V, then
  % layers{V} = L.
  assert(~isempty(net.vars)) ;
  layers = cell(size(net.vars)) ;
  
  % create Input layers
  inputNames = fieldnames(net.inputs) ;
  
  for k = 1:numel(inputNames)
    var = net.inputs.(inputNames{k}) ;  % the var index
    layers{var} = Input('name', inputNames{k}, 'gpu', net.isGpuVar(var)) ;
  end
  
  % create Param layers
  for k = 1:numel(net.params)
    p = net.params(k) ;
    
    layers{p.var} = Param('name', p.name, 'value', net.vars{p.var}, ...
      'gpu', net.isGpuVar(p.var), 'learningRate', p.learningRate, ...
      'weightDecay', p.weightDecay, 'trainMethod', p.trainMethod) ;
  end
  
  % recursively process the vars, starting with the last one (root)
  rootLayer = convertLayer(numel(net.vars) - 1) ;
  
  if ~isequal(rootLayer.func, @root)
    % single output
    netOutputs = {rootLayer} ;
  else
    % root layer's arguments are the network's outputs
    netOutputs = rootLayer.inputs ;
  end
  
  % copy meta properties to one of the Layers
  netOutputs{1}.meta = net.meta ;
  
  
  % process a single layer, and recurse on its inputs
  function obj = convertLayer(varIdx)
    % if this layer has already been processed, return its handle
    if ~isempty(layers{varIdx})
      obj = layers{varIdx} ;
      return
    end
    
    layer = forward(var2forward(varIdx)) ;
    args = layer.args ;
    
    % recurse on input Layers; they must be defined before this one.
    % this fills in the args list with the resulting Layer objects.
    for i = 1:numel(layer.inputVars)
      args{layer.inputArgPos(i)} = convertLayer(layer.inputVars(i)) ;
    end
    
    % now create a Layer with those arguments
    obj = Layer(layer.func, args{:}) ;
    
    % retrieve the number of input derivatives from the backward struct
    obj.numInputDer = backward(end - var2forward(varIdx) + 1).numInputDer ;
    
    % store in the 'layers' cell array so other references to the same var
    % will fetch the same layer
    obj.name = layer.name ;
    layers{layer.outputVar(1)} = obj ;
    
    % if the layer has multiple outputs, create a Selector for each of the
    % outputs after the first one
    for i = 2:numel(layer.outputVar)
      layers{layer.outputVar(i)} = Selector(obj, i) ;
    end
  end
end

