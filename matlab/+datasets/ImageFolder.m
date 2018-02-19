classdef ImageFolder < datasets.Dataset
  %ImageFolder Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    dataDir
    filenames
    
    numThreads = 1  % higher to enable multithreading/prefetching
    useGpu = false
    removeMean = true
    whiten = true
    
    imageSize
    keepAspect = true

    augmentation = struct('flip', false, 'location', false, 'aspect', 1, ...
      'scale', 1, 'brightness', 0, 'contrast', 0, 'saturation', 0, 'crop', 1)
    
    augmentImage  % vector, for each image, false disables augmentation (e.g. val images)
    
    % pixel statistics over dataset (computed automatically)
    rgbMean
    rgbCovariance
    rgbDeviation  % independent standard deviation of each channel (3x1)
    rgbDeviationFull  % standard deviation as a 3x3 matrix
    initialized = false
  end
  
  methods
    function o = ImageFolder(varargin)
      % parse generic ImageFolder arguments on non-subclassed
      % construction. allows use as a stand-alone class.
      varargin = o.parseGenericArgs(varargin) ;
      
      opts.skipInitialization = false ;
      opts = vl_argparse(opts, varargin) ;
      
      if ~opts.skipInitialization
        o.initialize() ;
      end
    end
    
    function initialize(o)
      % finishes initializing the dataset, must be called by subclasses
      assert(~o.initialized) ;
      assert(~isempty(o.dataDir), 'Must specify dataset.dataDir.') ;
      assert(~isempty(o.imageSize), 'Must specify dataset.imageSize.') ;
      o.initialized = true ;
      
      % list image files if they weren't set
      if isempty(o.filenames)
        o.filenames = o.listImages(o.dataDir) ;
      end
      
      % apply data augmentation to all images by default
      if isempty(o.augmentImage)
        o.augmentImage = true(size(o.filenames)) ;
      end
      
      % load or recompute RGB statistics over all images
      if o.removeMean || o.whiten
        cache = [o.dataDir '/datasetstats.mat'] ;
        if exist(cache, 'file')
          s = load(cache) ;
          o.rgbMean = s.rgbMean ;
          o.rgbCovariance = s.rgbCovariance ;
        else
          o.computeImageStats() ;
          rgbMean = o.rgbMean ;  %#ok<*PROP,NASGU>
          rgbCovariance = o.rgbCovariance ;  %#ok<NASGU>
          save(cache, 'rgbMean', 'rgbCovariance') ;
        end
        
        % standard deviation
        o.rgbDeviation = sqrt(diag(o.rgbCovariance)) ;
        [v, d] = eig(o.rgbCovariance) ;
        o.rgbDeviationFull = v * sqrt(d) ;
      end
      
      % reset MEX file state
      clear('vl_imreadjpeg') ;
    end
    
    function files = listImages(o, folder, prefix)
      % returns all JPEG image files recursively from the given directory
      
      if nargin < 3  % no leading slash on top-level prefix/partial folder
        prefix = '' ;
      else
        prefix = [prefix '/'] ;
      end
      folder = [folder '/'] ;
      
      % list folder contents
      s = dir([folder '*']) ;
      
      % iterate files/folders
      files = cell(1, numel(s)) ;
      for i = 1:numel(s)
        name = s(i).name ;
        if strcmpi(name(max(1,end-3) : end), '.jpg') || ...
           strcmpi(name(max(1,end-4) : end), '.jpeg')
          % a file, store it (wrapped around a scalar cell)
          files{i} = {[prefix name]} ;
          
        elseif s(i).isdir && ~any(strcmp(name, {'.', '..'}))
          % a folder, recurse (stored as a cell array)
          files{i} = o.listImages([folder name], [prefix name]) ;
        end
      end
      
      % flatten nested cell arrays. this also removes empty entries, which
      % were not assigned in the For loop (e.g. invalid files/folders).
      files = [files{:}] ;
    end
    
    function args = parseGenericArgs(o, args)
      % called by subclasses to parse generic Dataset arguments.
      % start by parsing parent class's arguments
      args = o.parseGenericArgs@datasets.Dataset(args) ;
      
      % now parse ImageFolder arguments
      args = vl_parseprop(o, args, {'dataDir', 'filenames', 'removeMean', ...
        'whiten', 'numThreads', 'useGpu', 'imageSize', 'keepAspect', ...
        'augmentation'}) ;
    end
    
    function batches = partition(o, idx, batchSz)
      % partition indexes into batches (stored in a cell array).
      % if IDX is a matrix, each column is a distinct sample.
      if nargin < 3  % allow overriding batch size
        batchSz = o.batchSize ;
      end
      batches = o.partition@datasets.Dataset(idx, batchSz) ;  % call parent class
      
      % for prefetching, append an extra row with the next batch for each
      % batch. this way, get() can know which batch comes next to prefetch.
      if o.numThreads > 1
        batches = [batches; batches(1,2:end), {[]}] ;
      end
    end
    
    function [images, idx] = get(o, batch)
      % return a single batch of images, with optional prefetching.
      % also returns the image indexes.
      if ~iscell(batch)
        batch = {batch};
      end
      assert(~isempty(batch) && numel(batch) <= 2, ...
        'Malformed batch; use dataset.partition/train/val.');
      
      % get current batch
      idx = batch{1} ;
      images = o.getImageBatch(o.filenames(idx), o.augmentImage(idx(1))) ;
      
      % start prefetching next batch of images
      if o.numThreads > 1 && numel(batch) == 2 && ~isempty(batch{2})
        o.getImageBatch(o.filenames(batch{2}), o.augmentImage(batch{2}(1))) ;
      end
    end
    
    function data = getImageBatch(o, images, augmentImages)
      % load image batch, with data augmentation and whitening
      assert(o.initialized, 'Subclass did not call ImageFolder.initialize.') ;

      imagePaths = strcat([o.dataDir filesep], images) ;
      
      % for validation, disable all augmentation except any fixed cropping
      if augmentImages
        augment = o.augmentation ;  % full augmentation
      else
        augment = struct('crop', o.augmentation.crop) ;  % crop only
      end
      
      % fill in missing augmentation options with defaults
      defaults = struct('flip', false, 'location', false, 'aspect', 1, ...
        'scale', 1, 'brightness', 0, 'contrast', 0, 'saturation', 0, 'crop', 1) ;
      augment = vl_argparse(defaults, augment) ;
      
      % make brightness jitter proportional to dataset's color deviation,
      % unless the data is whitened
      if ~isequal(augment.brightness, 0) && ~o.whiten
        if ~isempty(o.rgbDeviationFull)
          augment.brightness = double(augment.brightness * o.rgbDeviationFull)  ;
        else  % no color deviation to use
          augment.brightness = 0 ;
        end
      end
      
      args{1} = {imagePaths, ...
                 'NumThreads', o.numThreads, ...
                 'Pack', ...
                 'Interpolation', 'bicubic', ...
                 'Resize', o.imageSize(1:2), ...
                 'CropSize', augment.crop * augment.scale, ...
                 'CropAnisotropy', augment.aspect, ...
                 'Brightness', augment.brightness, ...
                 'Contrast', augment.contrast, ...
                 'Saturation', augment.saturation} ;

      if ~o.keepAspect  % Squashing effect
        args{end+1} = {'CropAnisotropy', 0} ;
      end

      if augment.flip
        args{end+1} = {'Flip'} ;
      end

      if augment.location
        args{end+1} = {'CropLocation', 'random'} ;
      else
        args{end+1} = {'CropLocation', 'center'} ;
      end

      if o.useGpu
        args{end+1} = {'Gpu'} ;
      end

      if o.removeMean && ~isempty(o.rgbMean)
        args{end+1} = {'SubtractAverage', o.rgbMean} ;
      end

      args = horzcat(args{:}) ;

      if o.numThreads > 1 && nargout == 0
        vl_imreadjpeg(args{:}, 'prefetch') ;
      else
        data = vl_imreadjpeg(args{:}) ;
        data = data{1} ;
        
        if o.whiten && ~isempty(o.rgbDeviation)
          data = data ./ reshape(o.rgbDeviation, 1, 1, 3) ;
        end
      end
    end

    function computeImageStats(o)
      % computes RGB mean/covariance over whole dataset
      bs = o.batchSize ;
      images = o.filenames ;
      [avg, rgbm1, rgbm2] = deal({}) ;

      maxImages = 10000 ;  % max number of images, should be enough for pixel-wise mean
      skip = floor(max(1, numel(images) / maxImages)) ;  % number of batches to skip
      for t = 1 : skip * bs : numel(images)
        time = tic ;
        batch = t : min(t+bs-1, numel(images)) ;
        fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;

        data = o.getImageBatch(images(batch), false) ;

        z = reshape(shiftdim(data,2),3,[]) ;
        rgbm1{end+1} = mean(z,2) ;
        rgbm2{end+1} = z*z'/size(z,2) ;
        avg{end+1} = mean(data, 4) ;
        time = toc(time) ;
        fprintf(' %.1f Hz\n', numel(batch) / time) ;
      end

      % averageImage = gather(mean(cat(4,avg{:}),4)) ;
      rgbm1 = gather(mean(cat(2,rgbm1{:}),2)) ;
      rgbm2 = gather(mean(cat(3,rgbm2{:}),3)) ;
      o.rgbMean = rgbm1 ;
      o.rgbCovariance = rgbm2 - rgbm1*rgbm1' ;

      fprintf('computeImageStats: all done\n') ;
    end
  end
end

