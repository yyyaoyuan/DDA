function d = checkgrad(f, X, e, varargin)

% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad('f', X, e,  varargin{:})
%
% where X is the argument and e is the small perturbation used for the finite
% differences. and varargin contains optional additional parameters which
% get passed to f. The function f should be of the type 
%
% [fX, dfX] = f(X, varargin{:})
%
% where fX is the function value and dfX is a vector of partial derivatives.
%

[y dy] = feval(f, X, varargin{:});     % get the partial derivatives dy

dh = zeros(length(X),1) ;
for j = 1:length(X)
  dx = zeros(length(X),1);
  dx(j) = e;                           % perturb a single dimension
  y2 = feval(f, X+dx, varargin{:});
  y1 = feval(f, X-dx, varargin{:});
  dh(j) = (y2 - y1)/(2*e);
end

disp([dy dh])                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);           % return norm of diff divided by norm of sum