function setParameterServer(net, ps)
%SETPARAMETERSERVER  Set a parameter server for the parameter derivatives
%    SETPARAMETERSERVER(obj, PS) uses the specified
%    ParameterServer PS to store and accumulate parameter
%    derivatives across multiple MATLAB processes.
%
%    After setting this option, the parameter derivatives are always empty
%    and must be retrieved from the server.

  % get initial values of all parameters
  paramValues = net.vars([net.params.var]) ;
  
  % register each one
  for p = 1:numel(net.params)
    if isa(paramValues{p}, 'gpuArray')
      deviceType = 'gpu' ;
      dataType = classUnderlying(paramValues{p}) ;
    else
      deviceType = 'cpu' ;
      dataType = class(paramValues{p}) ;
    end
    % use sequential names, net.params(p).name not guaranteed to be valid
    name = sprintf('p%i', p) ;
    ps.register(name, size(paramValues{p}), dataType, deviceType) ;
  end
  
  net.parameterServer = ps ;
  net.parameterServer.start() ;
end


