
function cifar_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = '/data/cifar' ;  % CIFAR10 data location
  opts.resultsDir = 'data/cifar-example' ;
  opts.numEpochs = 100 ;
  opts.batchSize = 128 ;
  opts.learningRate = 0.001 ;
  opts.gpu = 1 ;  % GPU index, empty for CPU mode
  opts.savePlot = false ;
  opts = vl_argparse(opts, varargin) ;
  
  try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
  mkdir(opts.resultsDir) ;
  

  % define network
  images = Input('gpu', true) ;
  labels = Input() ;

  x = vl_nnconv(images, 'size', [5, 5, 3, 32], 'pad', 2, 'weightScale', 0.01) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'max', 'pad', 1) ;
  x = vl_nnrelu(x) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 32, 32], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnrelu(x) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = vl_nnconv(x, 'size', [5, 5, 32, 64], 'pad', 2, 'weightScale', 0.05) ;
  x = vl_nnrelu(x) ;
  x = vl_nnpool(x, 3, 'stride', 2, 'method', 'avg', 'pad', 1) ;
  
  x = vl_nnconv(x, 'size', [4, 4, 64, 64], 'weightScale', 0.05) ;
  x = vl_nnrelu(x) ;
  
  x = vl_nnconv(x, 'size', [1, 1, 64, 10], 'weightScale', 0.05) ;

  objective = vl_nnloss(x, labels, 'loss', 'softmaxlog') / opts.batchSize ;
  error = vl_nnloss(x, labels, 'loss', 'classerror') / opts.batchSize ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, error) ;


  % initialize solver
  solver = solvers.SGD('learningRate', opts.learningRate) ;
  
  % initialize dataset
  dataset = datasets.CIFAR10(opts.dataDir, 'batchSize', opts.batchSize) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % continue from last checkpoint if there is one
  [filename, startEpoch] = get_last_checkpoint([opts.resultsDir '/epoch-*.mat']) ;
  if ~isempty(filename)
    load(filename, 'net', 'stats', 'solver') ;
  end

  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = startEpoch : opts.numEpochs
    % training phase
    for batch = dataset.train()
      % draw samples
      [images, labels] = dataset.get(batch) ;

      % evaluate network to compute gradients
      tic;
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;

      % get current objective and error, and update their average.
      % also report timing.
      fprintf('train - %.1fms ', toc() * 1000) ;
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

      fprintf('val - %.1fms ', toc() * 1000) ;
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

