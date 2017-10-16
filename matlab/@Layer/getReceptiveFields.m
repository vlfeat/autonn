function [rfSize, rfOffset, rfStride] = getReceptiveFields(obj, input, customRF)
%GETRECEPTIVEFIELDS Computes the receptive fields of a CNN
%   [RFSIZE, RFOFFSET, RFSTRIDE] = OBJ.GETRECEPTIVEFIELDS()
%   Returns the receptive fields of a single-stream CNN with output OBJ.
%
%   Because receptive fields are assumed to be rectangular, it only makes
%   sense to model them for CNNs (composed of convolutional and
%   element-wise operators only).
%
%   Unknown layers are assumed to be element-wise. It is also assumed that
%   all layers are composed using the first argument only. Unless specified
%   otherwise, the CNN is assumed to start with an Input layer. Taken
%   together these mean that the CNN has a single stream (it is a
%   sequential network). See below on how to override these behaviors.
%
%   All 3 returned values are N-by-2 matrices, where N is the number of
%   layers in the stream. They contain for each layer the size, offset and
%   stride that define a rectangular receptive field.
%
%   They can be interpreted as follows. Given a pixel of vertical
%   coordinate u in an output variable OUT(y,...) , the first and last
%   pixels affecting that pixel in an input variable IN(v,...) are:
%
%     v_first = rfstride(L,1) * (y - 1) + rfoffset(L,1) - rfsize(L,1)/2 + 1
%     v_last  = rfstride(L,1) * (y - 1) + rfoffset(L,1) + rfsize(L,1)/2 + 1
%
%   And likewise (using index (L,2) instead of (L,1)) for the horizontal
%   coordinate. See the MatConvNet Manual (PDF) for a more detailed
%   exposition.
%
%   [...] = OBJ.GETRECEPTIVEFIELDS(INPUT)
%   Defines a different layer as the beginning (input) of the CNN stream.
%
%   [...] = OBJ.GETRECEPTIVEFIELDS(INPUT, CUSTOMRF)
%   Specifies a custom function to handle unknown layers (i.e., define the
%   receptive fields of custom layers). The function signature is:
%
%     [kernelsize, offfset, stride] = customRF(obj)
%
%   where obj is the unknown layer, and the returned values are the kernel
%   size, offset and stride respectively.
%
%   [...] = GETRECEPTIVEFIELDS({LAYER1, LAYER2, ...}, ...)
%   Defines a different sequential stream of layers {LAYER1, LAYER2, ...}.
%   The layers must be in forward order.
%
%   Joao F. Henriques, 2017

  if nargin < 2
    input = [] ;
  end
  if nargin < 3
    customRF = [] ;
  end
  
  % note: could improve error checking here.
  % to do: generalize to whole graph instead of single stream (see
  % DagNN.getVarReceptiveFields).
  if iscell(obj)
    % passed in a sequential list of layers
    layers = obj;
  else
    % traverse backwards on first input (assume single-stream CNN structure)
    layers = {} ;
    while ~isa(obj, 'Input') && ~eq(obj, input, 'sameInstance')
      layers{end+1} = obj ;
      if isa(obj.inputs{1}, 'Layer')
        obj = obj.inputs{1} ;
      else
        break
      end
    end
    
    % change to forward-order
    layers = layers(end:-1:1) ;
  end
  
  % allocate memory
  n = numel(layers) ;
  rfSize = zeros(2, n) ;
  rfOffset = zeros(2, n) ;
  
  support = zeros(2, n) ;
  stride = zeros(2, n) ;
  pad = zeros(4, n) ;
  
  % generic options to parse
  defaults.dilate = 1 ;
  defaults.stride = 1 ;
  defaults.pad = 0 ;
  
  % convolution-transpose options to parse
  convtDefaults.upsample = 1 ;
  convtDefaults.crop = 0 ;
  
  for i = 1:numel(layers)
    l = layers{i} ;
    
    [opts, ~] = vl_argparsepos(defaults, l.inputs, 'merge') ;
    
    stride(:,i) = opts.stride(:) ;
    pad(:,i) = opts.pad(:) ;
    
    switch func2str(l.func)
    case 'vl_nnconv'
      % convolution
      support(:,i) = getKernelSize(l, opts.dilate) ;
      
    case 'vl_nnpool'
      % pooling
      assert(isnumeric(l.inputs{2}), 'Pooling size is not numeric.') ;
      support(:,i) = l.inputs{2} ;
      
    case 'vl_nnconvt'
      % convolution-transpose
      [opts, ~] = vl_argparsepos(convtDefaults, l.inputs, 'merge') ;
      
      ks = getKernelSize(l, 1) ;
      
      support(:,i) = (ks - 1) ./ opts.upsample + 1 ;
      stride(:,i) = 1 ./ [opts.upsample] ;
      pad(:,i) = (2 * opts.crop([1 3]) - ks + 1) ./ (2 * opts.upsample) + 1 ;
      
    otherwise
      % others, assume element-wise by default
      support(:,i) = 1 ;
      
      if ~isempty(customRF)
        % use handler for custom layers
        [ks, of, st] = customRF(l);  %#ok<RHSFN>
        if ~isempty(ks)
          support(:,i) = ks ;
        end
        if ~isempty(of)
          pad(:,i) = of ;
        end
        if ~isempty(st)
          stride(:,i) = st ;
        end
      end
    end

    % operator applied to the input image
    rfSize(:,i) = 1 + sum(cumprod([[1 ; 1], stride(:, 1:i-1)], 2) .* ...
      (support(:, 1:i) - 1), 2) ;
    
    rfOffset(:,i) = 1 + sum(cumprod([[1 ; 1], stride(:, 1:i-1)], 2) .* ...
      ((support(:, 1:i) - 1) / 2 - pad([1 3], 1:i)), 2) ;
  end

  rfStride = cumprod(stride, 2) ;
end

function ks = getKernelSize(l, dilate)
  % obtains the kernel size, taking dilation into account.
  % assumes L is a convolution/conv-transpose layer.
  assert(isa(l.inputs{2}, 'Param')) ;
  w = l.inputs{2}.value ;

  ks = max([size(w,1); size(w,2)], 1) ;
  ks = (ks - 1) .* dilate + 1 ;
end

