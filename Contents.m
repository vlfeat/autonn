% AutoNN - Table of Contents
%
% Classes
%   Layer              - Main building block for defining new networks
%   Net                - Compiled network that can be evaluated on data
%   Input              - Defines a network input (such as images or labels)
%   Param              - Defines a learnable network parameter
%   Selector           - Selects a single output of a multiple-outputs layer
%   Var                - Defines a network variable explicitly
%
% Layer methods
%   display            - Display layer information
%   find               - Searches for layers that meet the specified criteria
%   deepCopy           - Copies a network or subnetwork, optionally sharing some layers
%   evalOutputSize     - Computes output size of a layer
%   plotPDF            - Displays the network topology in a PDF
%   workspaceNames     - Sets names of unnamed layers based on the current workspace
%   sequentialNames    - Sets names of all unnamed layers based on type and order
%
% Layer overloaded methods
%   Operators          - Many functions and operators are overloaded, see: methods('Layer')
%   MatConvNet layers  - All MatConvNet layer functions are overloaded, see: methods('Layer')
%   vl_nnconv          - Additional options for vl_nnconv (CNN convolution)
%   vl_nnconvt         - Additional options for vl_nnconvt (CNN deconvolution)
%   vl_nnbnorm         - Additional options for vl_nnbnorm (CNN batch normalisation)
%   vl_nndropout       - Additional options for vl_nndropout (CNN dropout)
%   eq                 - Overloaded equality operator, or test for Layer instance equality
%
% Layer static methods
%   fromDagNN          - Converts a DagNN object to the AutoNN framework
%   fromCompiledNet    - Decompiles a Net back into Layer objects
%   fromFunction       - Generator for new custom layer type
%   create             - Creates a layer from a function handle and arguments
%
% Net methods
%   eval               - Network evaluation, including backpropagation to compute derivatives
%   displayVars        - Display table with information on variables and derivatives
%   getVarsInfo        - Retrieves network variables information as a struct
%   plotDiagnostics    - Creates or updates diagnostics plot
%   setParameterServer - Sets up a parameter server for multi-GPU training
%
% Extra CNN blocks
%   vl_nnlstm          - Long Short-Term Memory cell (LSTM)
%   vl_nnlstm_params   - Initialize the learnable parameters for an LSTM
%   vl_nnaffinegrid    - Affine grid generator for Spatial Transformer Networks
%   vl_nnmask          - CNN dropout mask generator
%
% Utilities
%   setup_autonn       - Sets up AutoNN, by adding its folders to the Matlab path
%   cnn_train_autonn   - Demonstrates training a CNN using the AutoNN wrapper
%   vl_argparsepos     - Parse list of param.-value pairs, with positional arguments
%   dynamic_subplot    - Dynamically reflowing subplots, to maintain aspect ratio
%
% Examples
%   examples/minimal   - Directory with minimal examples: regression and LSTM (start here)
%   examples/cnn       - Directory with CNN examples: ImageNet, MNIST and toy data
%   examples/rnn       - Directory with RNN/LSTM language model example on Shakespeare text
