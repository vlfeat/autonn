function unknown_args = vl_parseprop(obj, args, properties)
%VL_PARSEPROP Summary of this function goes here
%   Detailed explanation goes here

  assert(iscellstr(properties)) ;
  
  assert(mod(numel(args), 2) == 0, 'Expected name-value pairs.') ;
  
  names = args(1:2:end-1) ;
  values = args(2:2:end) ;
  assert(iscellstr(names), 'Expected name-value pairs.') ;
  
  unmatched = true(1, numel(names)) ;

  for i = 1:numel(properties)
    pos = find(strcmpi(names, properties{i}), 1, 'last') ;
    
    if isscalar(pos)
      obj.(properties{i}) = values{pos} ;
      unmatched(pos) = false ;
    end
  end
  
  if nargout == 0
    if any(unmatched)
      error('Unknown argument: %s', names{find(unmatched, 1)}) ;
    end
  else
    unknown_args = [names(unmatched) ; values(unmatched)] ;
    unknown_args = unknown_args(:)' ;
  end
end

