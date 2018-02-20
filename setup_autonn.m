function setup_autonn()
%SETUP_AUTONN Sets up AutoNN
%   SETUP_AUTONN sets up AutoNN, by adding its folders to the Matlab path.
%
%   Note that MatConvNet should also be on the path, by calling VL_SETUPNN.

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/matlab/wrappers'], [root '/matlab/derivatives']) ;
  
  if ~exist('vl_setupnn', 'file')
    warning('AutoNN:MatConvNetMissing', ['MatConvNet is not on the path. '...
      'You can set it up by calling:\n  run MATCONVNET/vl_setupnn\n' ...
      'replacing ''MATCONVNET'' with the directory where it is located.']) ;
  end

end

