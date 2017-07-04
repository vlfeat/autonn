classdef MNIST < datasets.Dataset
  %MNIST Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    images  % images tensor
    dataMean  % image mean
    labels  % labels (1 to 10)
    trainIdx  % indexes of training samples
    valIdx  % indexes of validation samples
  end
  
  methods
    function dataset = MNIST(dataDir, varargin)
      % parse generic Dataset arguments
      varargin = dataset.parseGenericArgs(varargin) ;
      assert(isempty(varargin), 'Unknown arguments.') ;
      
      % load object from cache if possible
      cache = [dataDir '/mnist.mat'] ;
      if exist(cache, 'file')
        load(cache, 'dataset') ;
      else
        dataset.loadRawData(dataDir) ;
        save(cache, 'dataset') ;
      end
    end
    
    function [images, labels] = get(o, idx)
      % return a single batch (may be wrapped in a cell)
      if iscell(idx) && isscalar(idx)
        idx = idx{1} ;
      end
      images = o.images(:,:,:,idx) ;
      labels = o.labels(idx) ;
    end
    
    function batches = train(o)
      % shuffle training set, and partition it into batches
      batches = o.partition(o.trainIdx(randperm(end))) ;
    end
    
    function batches = val(o)
      % partition validation set into batches
      batches = o.partition(o.valIdx) ;
    end
  end
  
  methods (Access = protected)
    function loadRawData(o, data_dir)
      % download and load raw MNIST data
      files = {'train-images-idx3-ubyte', ...
               'train-labels-idx1-ubyte', ...
               't10k-images-idx3-ubyte', ...
               't10k-labels-idx1-ubyte'} ;

      if ~exist(data_dir, 'dir')
        mkdir(data_dir) ;
      end

      for i=1:4
        if ~exist(fullfile(data_dir, files{i}), 'file')
          url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
          fprintf('downloading %s\n', url) ;
          gunzip(url, data_dir) ;
        end
      end

      f=fopen(fullfile(data_dir, 'train-images-idx3-ubyte'),'r') ;
      x1=fread(f,inf,'uint8') ;
      fclose(f) ;
      x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

      f=fopen(fullfile(data_dir, 't10k-images-idx3-ubyte'),'r') ;
      x2=fread(f,inf,'uint8') ;
      fclose(f) ;
      x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

      f=fopen(fullfile(data_dir, 'train-labels-idx1-ubyte'),'r') ;
      y1=fread(f,inf,'uint8') ;
      fclose(f) ;
      y1=double(y1(9:end)')+1 ;

      f=fopen(fullfile(data_dir, 't10k-labels-idx1-ubyte'),'r') ;
      y2=fread(f,inf,'uint8') ;
      fclose(f) ;
      y2=double(y2(9:end)')+1 ;
      
      o.trainIdx = 1 : numel(y1) ;
      o.valIdx = numel(y1) + 1 : numel(y1) + numel(y2) ;
      
      im = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
      o.dataMean = mean(im(:,:,:,o.trainIdx), 4) ;
      o.images = bsxfun(@minus, im, o.dataMean) ;
      o.labels = cat(2, y1, y2) ;
    end
  end
end

