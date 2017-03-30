function display(obj, name)
%DISPLAY Display layer information
%   OBJ.DISPLAY() overloads DISPLAY to show hyperlinks in command window,
%   allowing one to interactively traverse the network. Note that the
%   builtin DISP is unchanged.

  if nargin < 2
    name = inputname(1) ;
  end
  fprintf('\n%s', name) ;

  if builtin('numel', obj) ~= 1  % non-scalar, use standard display
    fprintf(' =\n\n') ;
    disp(obj) ;
    return
  end

  if numel(name) > 30, fprintf('\n'); end  % line break for long names
  showLinks = usejava('desktop') ;
  fprintf(' = ') ;
  obj.displayCustom(name, showLinks) ;

  if ~isempty(obj.source)  % show source-code origin
    [~, file, ext] = fileparts(obj.source(1).file) ;
    if ~showLinks
      fprintf('Defined in %s%s, line %i.\n', file, ext, obj.source(1).line) ;
    else
      fprintf('Defined in <a href="matlab:opentoline(''%s'',%i)">%s%s, line %i</a>.\n', ...
        obj.source(1).file, obj.source(1).line, file, ext, obj.source(1).line) ;
    end
  end

  if showLinks
    fprintf('(<a href="matlab:disp(%s)">Show all properties</a>)\n', name) ;
  else
    fprintf('(Use disp(%s) to show all properties)\n', name) ;
  end
end

