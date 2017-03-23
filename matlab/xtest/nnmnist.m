classdef nnmnist < nntest
  methods (TestClassSetup)
    function init(test)
      addpath([fileparts(mfilename('fullpath')) '/../../examples/cnn']);
    end
  end

  methods (Test)
    function valErrorRate(test)
%       clear mex ; % will reset GPU, remove MCN to avoid crashing
%                   % MATLAB on exit (BLAS issues?)
      if strcmp(test.currentDataType, 'double'), return ; end
      rng(0);  % fix random seed, for reproducible tests
      switch test.currentDevice
        case 'cpu'
          gpus = [];
        case 'gpu'
          gpus = 1;
      end
      trainOpts = struct('numEpochs', 1, 'continue', false, 'gpus', gpus, ...
        'plotStatistics', false, 'plotDiagnostics', false);
      
      [~, info] = cnn_mnist_autonn('train', trainOpts) ;
      
      test.verifyLessThan(info.train.error, 0.13);
      test.verifyLessThan(info.val.error, 0.11);
    end
  end
end
