function setup()
%SETUP Sets up autonn, by adding the relevant folders to the Matlab path.

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/wrappers'], [root '/derivatives']) ;

end

