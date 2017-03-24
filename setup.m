function setup()
%SETUP Sets up autonn, by adding the relevant folders to the Matlab path.

  root = fileparts(mfilename('fullpath')) ;
  addpath([root '/matlab'], [root '/matlab/wrappers'], [root '/matlab/derivatives']) ;

end

