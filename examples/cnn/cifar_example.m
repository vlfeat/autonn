
function cifar_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = [vl_rootnn() '/data/cifar'] ;  % CIFAR10 data location
  opts.resultsDir = [vl_rootnn() '/data/cifar-example'] ;  % results location
  opts.model = @models.BasicCifarNet ;  % choose model (type 'help models' for a list)
  opts.conserveMemory = true ;  % whether to conserve memory (helpful with e.g. @models.MaxoutNIN)
  opts.numEpochs = [] ;  % if empty, default for above model will be used
  opts.batchSize = [] ;  % same as above
  opts.learningRate = [] ;  % same as above
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  opts.savePlot = false ;  % whether to save the plot as a PDF file
  opts.continue = true ;  % continue from last checkpoint if available
  
  opts = vl_argparse(opts, varargin) ;  % let user override options
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % create network inputs
  images = Input('gpu', true) ;
  labels = Input() ;

  % create network specified in opts.model (from 'autonn/matlab/+models/')
  [output, defaults] = opts.model('input', images) ;
  
  % replace empty options with the model-specific default values
  if isempty(opts.numEpochs)
    opts.numEpochs = defaults.numEpochs ;
  end
  if isempty(opts.batchSize)
    opts.batchSize = defaults.batchSize ;
  end
  if isempty(opts.learningRate)
    opts.learningRate = defaults.learningRate ;
  end

  % create losses
  objective = vl_nnloss(output, labels, 'loss', 'softmaxlog') / opts.batchSize ;
  error = vl_nnloss(output, labels, 'loss', 'classerror') / opts.batchSize ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, error, 'conserveMemory', opts.conserveMemory) ;


  % initialize solver
  solver = solvers.SGD('learningRate', opts.learningRate(1)) ;
  
  % initialize dataset
  dataset = datasets.CIFAR10(opts.dataDir, 'batchSize', opts.batchSize) ;
  
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

