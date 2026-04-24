% gradient descent algorithm.

function [xs,fs,k] = grad_desc(fname,gname,x0,D,mu,n) % n is k in main
format compact
format long

k = 1;
xk = x0;
gk = feval(gname,xk,D,mu);
dk = -gk;
ak = bt_lsearch2019(xk,dk,fname,gname,D,mu); % D and mu unused, helper function not in repo

adk = ak*dk;
er = norm(adk);

while k < n
    xk = xk + adk;
    gk = feval(gname,xk,D,mu);
    dk = -gk;
    ak = bt_lsearch2019(xk,dk,fname,gname,D,mu);
    adk = ak*dk;
    er = norm(adk);
    k = k + 1;
end

disp('solution:')
xs = xk + adk

disp('objective function at solution point:')
fs = feval(fname,xs,D,mu)

format short
disp('number of iterations performed:')
k
