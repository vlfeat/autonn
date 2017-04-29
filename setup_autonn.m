function setup_autonn()
%SETUP_AUTONN Sets up AutoNN, by adding its folders to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/matlab/wrappers'], [root '/matlab/derivatives']) ;

end

