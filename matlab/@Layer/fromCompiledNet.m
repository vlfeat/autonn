function netOutputs = fromCompiledNet(net)
%FROMCOMPILEDNET Converts a compiled Net object to a cell array 
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

  % handles of all created Layer handles, empty if not processed.
  % executed layers go first, followed by Inputs, then Params.
  layers = cell(1, numel(forward) + numel(fieldnames(net.inputs)) + numel(net.params)) ;

  % mapping between each layer and its output var
  assert(~isempty(net.vars)) ;
  var2layer = zeros(size(net.vars)) ;
  
  % set up mapping for executed layers
  for k = 1:numel(forward)
    assert(isscalar([forward(k).outputVar]), ...  % TO DO: handle this case
      'Converting layers with multiple outputs is not supported yet.') ;
    
    var2layer(forward(k).outputVar) = k ;
  end
  
  % set up mapping for Input layers, and create them
  inputNames = fieldnames(net.inputs) ;
  offset = numel(forward) ;
  
  for k = 1:numel(inputNames)
    var = net.inputs.(inputNames{k}) ;  % the var index
    
    layers{offset + k} = Input('name', inputNames{k}, 'gpu', net.isGpuVar(var)) ;
    
    var2layer(var) = offset + k ;
  end
  
  % set up mapping for Param layers, and create them
  offset = numel(forward) + numel(fieldnames(net.inputs)) ;
  
  for k = 1:numel(net.params)
    p = net.params(k) ;
    
    layers{offset + k} = Param('name', p.name, 'value', net.vars{p.var}, ...
      'gpu', net.isGpuVar(p.var), 'learningRate', p.learningRate, ...
      'weightDecay', p.weightDecay, 'trainMethod', p.trainMethod) ;
    
    var2layer(p.var) = offset + k ;
  end
  
  % recursively process the layers, starting with the last one (root)
  rootLayer = convertLayer(numel(forward)) ;
  
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
  function obj = convertLayer(layerIdx)
    % if this layer has already been processed, return its handle
    if ~isempty(layers{layerIdx})
      obj = layers{layerIdx} ;
      return
    end
    
    layer = forward(layerIdx) ;
    args = layer.args ;
    
    % recurse on input Layers; they must be defined before this one.
    % this fills in the args list with the resulting Layer objects.
    for i = 1:numel(layer.inputVars)
      args{layer.inputArgPos(i)} = convertLayer(var2layer(layer.inputVars(i))) ;
    end
    
    % now create a Layer with those arguments
    obj = Layer(layer.func, args{:}) ;
    
    % retrieve the number of input derivatives from the backward struct
    obj.numInputDer = backward(end - layerIdx + 1).numInputDer ;
    
    obj.name = layer.name ;
    
    layers{layerIdx} = obj ;
  end
end
