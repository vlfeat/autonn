function [filename, epoch] = get_last_checkpoint(pattern)
%GET_LAST_CHECKPOINT Summary of this function goes here
%   Detailed explanation goes here

  assert(nnz(pattern == '*') == 1, 'The file pattern must include exactly one wildcard (*).') ;
  
  % gather files respecting pattern
  files = dir(pattern) ;
  
  % replace wildcard with regexp to extract an integer from each file name
  [~, file_pattern] = fileparts(pattern);
  iterations = regexp({files.name}, strrep(file_pattern, '*', '([\d]+)'), 'tokens');
  
  % convert string integer to numeric, from doubly-nested cells
  iterations = cellfun(@(x) sscanf(x{1}{1}, '%d'), iterations);
  
  if isempty(iterations)
    % no files found
    filename = [] ;
    epoch = 1 ;
  else
    % return last file name, and starting epoch
    filename = strrep(pattern, '*', int2str(max(iterations))) ;
    epoch = max(iterations) + 1 ;
  end
end
