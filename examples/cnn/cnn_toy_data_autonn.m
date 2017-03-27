function [net, stats] = cnn_toy_data_autonn(varargin)
% CNN_TOY_DATA
% Minimal demonstration of MatConNet training of a CNN on toy data.
%
% It also serves as a short tutorial on creating and using a custom imdb
% (image database).
%
% The task is to distinguish between images of triangles, squares and
% circles.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

run('../../setup_autonn.m') ;  % add AutoNN to the path

% Parameter defaults. You can add any custom parameters here (e.g.
% opts.alpha = 1), and change them when calling: cnn_toy_data('alpha', 2).
opts.train.batchSize = 200 ;
opts.train.numEpochs = 10 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.01 ;
opts.train.expDir = [vl_rootnn '/data/toy'] ;
opts.dataDir = [vl_rootnn '/data/toy-dataset'] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = [opts.train.expDir '/imdb.mat'] ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% Generate images if they don't exist (this would be skipped for real data)
if ~exist(opts.dataDir, 'dir')
  addpath([vl_rootnn '/examples/custom_imdb']) ;
  mkdir(opts.dataDir) ;
  cnn_toy_data_generator(opts.dataDir) ;
end

% Create image database (imdb struct). It can be cached to a file for speed
if exist(opts.imdbPath, 'file')
  disp('Reloading image database...')
  imdb = load(opts.imdbPath) ;
else
  disp('Creating image database...')
  imdb = getImdb(opts.dataDir) ;
  mkdir(fileparts(opts.imdbPath)) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


% Create network by composing layers/functions, starting with the inputs
images = Input('gpu', true) ;  % Automatically move images to the GPU if needed
labels = Input() ;

f = 1/100 ;
x = vl_nnconv(images, 'size', [5, 5, 1, 5], 'weightScale', f) ;
x = vl_nnpool(x, 2, 'stride', 2) ;
x = vl_nnconv(x, 'size', [5, 5, 5, 10], 'weightScale', f) ;
x = vl_nnpool(x, 2, 'stride', 2) ;
x = vl_nnconv(x, 'size', [5, 5, 10, 3], 'weightScale', f) ;

objective = vl_nnloss(x, labels) ;  % What we minimize
error = vl_nnloss(x, labels, 'loss', 'classerror') ;  % The error metric

Layer.workspaceNames() ;  % Assign layer names based on workspace variables (e.g. 'images', 'objective')
net = Net(objective, error) ;  % Compile network


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Start training
[net, stats] = cnn_train_autonn(net, imdb, @getBatch, ...
  'train', find(imdb.set == 1), 'val', find(imdb.set == 2), opts.train) ;

% Visualize the learned filters
figure(3) ; vl_tshow(net.getValue('conv1_filters')) ; title('Conv1 filters') ;
figure(4) ; vl_tshow(net.getValue('conv2_filters')) ; title('Conv2 filters') ;
figure(5) ; vl_tshow(net.getValue('x_filters')) ; title('Conv3 filters') ;


% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------
% This is where we return a given set of images (and their labels) from
% our imdb structure.
% If the dataset was too large to fit in memory, getBatch could load images
% from disk instead (with indexes given in 'batch').

images = imdb.images(:,:,:,batch) ;
labels = imdb.labels(batch) ;

inputs = {'images', images, 'labels', labels} ;

% --------------------------------------------------------------------
function imdb = getImdb(dataDir)
% --------------------------------------------------------------------
% Initialize the imdb structure (image database).
% Note the fields are arbitrary: only your getBatch needs to understand it.
% The field imdb.set is used to distinguish between the training and
% validation sets, and is only used in the above call to cnn_train_autonn.

% The sets, and number of samples per label in each set
sets = {'train', 'val'} ;
numSamples = [1500, 150] ;

% Preallocate memory
totalSamples = 4950 ;  % 3 * 1500 + 3 * 150
images = zeros(32, 32, 1, totalSamples, 'single') ;
labels = zeros(totalSamples, 1) ;
set = ones(totalSamples, 1) ;

% Read all samples
sample = 1 ;
for s = 1:2  % Iterate sets
  for label = 1:3  % Iterate labels
    for i = 1:numSamples(s)  % Iterate samples
      % Read image
      im = imread(sprintf('%s/%s/%i/%04i.png', dataDir, sets{s}, label, i)) ;
      
      % Store it, along with label and train/val set information
      images(:,:,:,sample) = single(im) ;
      labels(sample) = label ;
      set(sample) = s ;
      sample = sample + 1 ;
    end
  end
end

% Show some random example images
figure(2) ;
montage(images(:,:,:,randperm(totalSamples, 100))) ;
title('Example images') ;

% Remove mean over whole dataset
images = bsxfun(@minus, images, mean(images, 4)) ;

% Store results in the imdb struct
imdb.images = images ;
imdb.labels = labels ;
imdb.set = set ;

