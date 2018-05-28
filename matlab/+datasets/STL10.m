classdef STL10 < datasets.Dataset
%STL10 STL-10 dataset
%   Encapsulates the STL-10 dataset for training.
%
%   D = datasets.STL10('/data/stl') loads the STL-10 dataset from the
%   directory '/data/stl'. The dataset can be downloaded from:
%   https://cs.stanford.edu/~acoates/stl10/
%
%   D.train() returns a cell array with mini-batches, from the shuffled
%   training set. Each mini-batch consists of a set of indexes.
%
%   D.val() returns a cell array with mini-batches, from the validation set
%   (without shuffling). Each mini-batch consists of a set of indexes.
%
%   [IMAGES, LABELS] = D.get(BATCH) returns a tensor of images and the
%   corresponding labels for the given mini-batch BATCH. The images are
%   always mean-centered (by subtracting D.dataMean).
%
%   datasets.STL10(...,'option', value, ...) sets the following properties:
%
%   `batchSize`:: 128
%     The batch size.
%
%   `partialBatches`:: false
%     Whether partial batches are returned (which can happen for the last
%     batch in a set, if the batch size does not divide the set size).
%
%   `small`:: false
%     Loads smaller images with size 48x48, instead of the original 96x96.

% Copyright (C) 2018 Joao F. Henriques, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    images  % images tensor
    dataMean  % image mean
    labels  % labels (1 to 10)
    labelNames  % string name associated with each numeric label
    sets  % set identities (1 for training, 3 for test)
    folds  % identities of original training folds as defined in the dataset
    
    small = false  % downsample images 2x
  end
  
  methods
    function o = STL10(dataDir, varargin)
      % parse generic Dataset arguments
      varargin = o.parseGenericArgs(varargin) ;
      
      % parse arguments specific to this dataset
      vl_parseprop(o, varargin, {'small'}) ;
      
      % load object from memory cache if possible
      persistent s
      if ~isempty(s) && isequal(o.small, s.small)
        [o.images, o.dataMean, o.labels, o.labelNames, o.sets, o.folds] = ...
          deal(s.images, s.dataMean, s.labels, s.labelNames, s.sets, s.folds);
        
      else
        % otherwise, create from scratch
        o.loadRawData(dataDir);

        % save to cache
        [s.images, s.dataMean, s.labels, s.labelNames, s.sets, s.folds, s.small] = ...
          deal(o.images, o.dataMean, o.labels, o.labelNames, o.sets, o.folds, o.small);
      end
    end
    
    function [images, labels] = get(o, idx)
      % return a single batch (may be wrapped in a cell)
      if iscell(idx) && isscalar(idx)
        idx = idx{1} ;
      end
      images = (single(o.images(:,:,:,idx)) - o.dataMean) / 255;
      labels = o.labels(idx);
    end
  end
  
  methods (Access = protected)
    function loadRawData(o, data_dir)
      % labeled training data
      s = load([data_dir '/train.mat']);
      data = convert(s.X);
      labels = s.y(:);
      classes = s.class_names;
      folds = s.fold_indices;

      % unlabeled training data
      s = load([data_dir '/unlabeled.mat']);
      data = cat(4, data, convert(s.X));
      labels = [labels; zeros(size(s.X, 1), 1)];

      set = ones(size(labels));

      % labeled test data
      s = load([data_dir '/test.mat']);
      data = cat(4, data, convert(s.X));
      labels = [labels; s.y(:)];
      set = [set; 3 * ones(size(s.X, 1), 1)];

      % resize by 1/2 with bilinear interpolation
      if o.small
        data = uint16(data);  % avoid overflow
        data = (data(1:2:end,1:2:end,:,:) + data(2:2:end,1:2:end,:,:) + ...
                data(1:2:end,2:2:end,:,:) + data(2:2:end,2:2:end,:,:)) / 4;
        data = uint8(data);
      end
      
      o.images = data;
      o.labels = labels;  %#ok<*PROPLC>
      o.labelNames = classes;
      o.sets = set;
      o.folds = folds;
      
      o.dataMean = single(mean(data(:,:,:,set == 1), 4));
      
      % train and validation sample indexes
      o.trainSet = find(set == 1);
      o.valSet = find(set == 3);
    end
  end
end

% simple helper function to convert STL-10 array to MatConvNet format
function X = convert(X)
  X = permute(reshape(X, [], 96, 96, 3), [2:4, 1]);
end
