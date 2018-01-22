
function imagenet_example(varargin)
  % options (override by calling script with name-value pairs).
  % (*) if left empty, the default value for the chosen model will be used.
  opts.dataDir = [vl_rootnn() '/data/ilsvrc12'] ;  % ImageNet data location
  opts.resultsDir = [vl_rootnn() '/data/imagenet-example'] ;  % results location
  opts.model = models.AlexNet() ;  % choose model (type 'help models' for a list)
  opts.conserveMemory = true ;  % whether to conserve memory
  opts.numEpochs = [] ;  % epochs (*)
  opts.batchSize = [] ;  % batch size (*)
  opts.learningRate = [] ;  % learning rate (*)
  opts.solver = solvers.SGD() ;  % solver instance to use (type 'help solvers' for a list)
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  opts.numThreads = 12 ;  % number of threads for image reading
  opts.augmentation = [] ;  % data augmentation (see datasets.StreamingDataset) (*)
  opts.savePlot = false ;  % whether to save the plot as a PDF file
  opts.continue = true ;  % continue from last checkpoint if available
  
  opts = vl_argparse(opts, varargin) ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % use chosen model's output as the predictions
  assert(isa(opts.model, 'Layer'), 'Model must be a CNN (e.g. models.AlexNet()).')
  predictions = opts.model ;
  
  % change the model's input name
  images = predictions.find('Input', 1) ;
  images.name = 'images' ;
  images.gpu = true ;
  
  % validate the prediction size (must predict 1000 classes)
  defaults = predictions.meta ;  % get model's meta information (default learning rate, etc)
  outputSize = predictions.evalOutputSize('images', [defaults.imageSize 5]) ;
  assert(isequal(outputSize, [1 1 1000 5]), 'Model output does not have the correct shape.') ;
  
  % replace empty options with model-specific default values
  for name_ = {'numEpochs', 'batchSize', 'learningRate', 'augmentation'}
    name = name_{1} ;  % dereference cell array
    if isempty(opts.(name))
      opts.(name) = defaults.(name) ;
    end
  end

  % create losses
  labels = Input() ;
  objective = vl_nnloss(predictions, labels, 'loss', 'softmaxlog') / opts.batchSize ;
  error = vl_nnloss(predictions, labels, 'loss', 'classerror') / opts.batchSize ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, error, 'conserveMemory', opts.conserveMemory) ;


  % set solver learning rate
  solver = opts.solver ;
  solver.learningRate = opts.learningRate(1)  ;
  
  % initialize dataset
  dataset = datasets.ImageNet('dataDir', opts.dataDir, 'imageSize', defaults.imageSize) ;
  dataset.batchSize = opts.batchSize ;
  dataset.augmentation = opts.augmentation ;
  dataset.numThreads = opts.numThreads ;
  dataset.useGpu = ~isempty(opts.gpu) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % continue from last checkpoint if there is one
  startEpoch = 1 ;
  if opts.continue
    [filename, startEpoch] = get_last_checkpoint([opts.resultsDir '/epoch-*.mat']) ;
  end
  if startEpoch > 1
    load(filename, 'net', 'stats', 'solver') ;
  end

  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = startEpoch : opts.numEpochs
    % get the learning rate for this epoch, if there is a schedule
    if epoch <= numel(opts.learningRate)
      solver.learningRate = opts.learningRate(epoch) ;
    end
    
    % training phase
    for batch = dataset.train()
      % draw samples
      [images, labels] = dataset.get(batch) ;
      
      % simple data augmentation: flip images horizontally
      if rand() > 0.5, images = fliplr(images) ; end
      
      % evaluate network to compute gradients
      tic;
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;

      % get current objective and error, and update their average.
      % also report iteration number and timing.
      fprintf('train %d - %.1fms ', stats.counts(1) + 1, toc() * 1000);
      stats.update(net) ;
      stats.print() ;
    end
    % push average objective and error (after one epoch) into the plot
    stats.push('train') ;

    % validation phase
    for batch = dataset.val()
      [images, labels] = dataset.get(batch) ;

      tic;
      net.eval({'images', images, 'labels', labels}, 'test') ;

      fprintf('val %d - %.1fms ', stats.counts(1) + 1, toc() * 1000);
      stats.update(net) ;
      stats.print() ;
    end
    stats.push('val') ;

    % plot statistics, with optional smoothing
    stats.plot('smoothen', 1) ;
    if opts.savePlot && ~isempty(opts.expDir)
      print(1, [opts.expDir '/plot.pdf'], '-dpdf') ;
    end
    
    % save checkpoint every few epochs
    if mod(epoch, 10) == 0
      save(sprintf('%s/epoch-%d.mat', opts.resultsDir, epoch), ...
        'net', 'stats', 'solver') ;
    end
  end

  % save results
  if ~isempty(opts.resultsDir)
    save([opts.resultsDir '/results.mat'], 'net', 'stats', 'solver') ;
  end
end

