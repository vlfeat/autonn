
function mnist_example(varargin)
  % options (override by calling script with name-value pairs)
  opts.dataDir = [vl_rootnn() '/data/mnist'] ;  % MNIST data location
  opts.resultsDir = [vl_rootnn() '/data/mnist-example'] ;
  opts.numEpochs = 20 ;
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

  x = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  x = vl_nnconv(x, 'size', [5, 5, 20, 50], 'weightScale', 0.01) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  x = vl_nnconv(x, 'size', [4, 4, 50, 500], 'weightScale', 0.01) ;
  x = vl_nnrelu(x) ;
  x = vl_nnconv(x, 'size', [1, 1, 500, 10], 'weightScale', 0.01) ;

  objective = vl_nnloss(x, labels, 'loss', 'softmaxlog') / opts.batchSize ;
  error = vl_nnloss(x, labels, 'loss', 'classerror') / opts.batchSize ;

  % assign layer names automatically, and compile network
  Layer.workspaceNames() ;
  net = Net(objective, error) ;


  % initialize solver
  solver = solvers.SGD('learningRate', opts.learningRate) ;
  
  % initialize dataset
  dataset = datasets.MNIST(opts.dataDir, 'batchSize', opts.batchSize) ;
  
  % compute average objective and error
  stats = Stats({'objective', 'error'}) ;
  
  % enable GPU mode
  net.useGpu(opts.gpu) ;

  for epoch = 1:opts.numEpochs
    % training phase
    for batch = dataset.train()
      % draw samples
      [images, labels] = dataset.get(batch) ;

      % evaluate network to compute gradients
      net.eval({'images', images, 'labels', labels}) ;
      
      % take one SGD step
      solver.step(net) ;

      % get current objective and error, and update their average
      stats.update(net) ;
      stats.print() ;
    end
    % push average objective and error (after one epoch) into the plot
    stats.push('train') ;

    % validation phase
    for batch = dataset.val()
      [images, labels] = dataset.get(batch) ;

      net.eval({'images', images, 'labels', labels}, 'test') ;

      stats.update(net) ;
      stats.print() ;
    end
    stats.push('val') ;

    % plot statistics
    stats.plot() ;
    if opts.savePlot && ~isempty(opts.expDir)
      print(1, [opts.expDir '/plot.pdf'], '-dpdf') ;
    end
  end

  % save results
  if ~isempty(opts.resultsDir)
    save([opts.resultsDir '/results.mat'], 'net', 'stats', 'solver') ;
  end
end

