% AUTONN
%
% Files
%   setup_autonn - Sets up AutoNN, by adding its folders to the Matlab path
%
% MATLAB
%
% Files
%   cnn_train_autonn - Demonstrates training a CNN using the AutoNN wrapper
%   dynamic_subplot  - Dynamically reflowing subplots, to maintain aspect ratio
%   Input            - Input Defines a network input (such as images or labels)
%   Param            - Param Defines a learnable network parameter
%   Selector         - Selector Selects a single output of a multiple-outputs layer
%   Var              - Var Defines a network variable explicitly
%   vl_argparsepos   - Parse list of param.-value pairs, with positional arguments
%   vl_nnaffinegrid  - Affine grid generator for Spatial Transformer Networks
%   vl_nnlstm        - Long Short-Term Memory cell (LSTM)
%   vl_nnlstm_params - Initialize the learnable parameters for an LSTM
%   vl_nnmask        - CNN dropout mask generator
%   vl_nnwsum        - Differentiable weighted sum
%
% @LAYER
%
% Files
%   create          - Creates a layer from a function handle and arguments
%   deepCopy        - Copies a network or subnetwork, optionally sharing some layers
%   display         - Display layer information
%   eq              - Overloaded equality operator, or test for Layer instance equality
%   evalOutputSize  - Computes output size of a Layer
%   find            - Searches for layers that meet the specified criteria
%   fromDagNN       - Converts a DagNN object to the AutoNN framework
%   fromFunction    - Define new custom layer
%   Layer           - Main building block for defining new networks
%   plotPDF         - Displays the network topology in a PDF
%   sequentialNames - Sets names of all unnamed layers based on type and order
%   vl_nnbnorm      - Overload for CNN batch normalisation
%   vl_nnconv       - Overload for CNN convolution
%   vl_nnconvt      - Overload for CNN convolution transpose
%   vl_nndropout    - Overload for CNN dropout
%   vl_nnwsum       - Overload for differentiable weighted sum
%   workspaceNames  - Sets names of unnamed layers based on the current workspace
%
% @NET
%
% Files
%   compile            - Compile network
%   displayVars        - Display table with information on variables and derivatives
%   eval               - Network evaluation, including back-propagation to compute derivatives
%   getVarsInfo        - Retrieves network variables information as a struct
%   Net                - Compiled network that can be evaluated on data
%   plotDiagnostics    - Creates or updates diagnostics plot
%   setParameterServer - Set a parameter server for the parameter derivatives
