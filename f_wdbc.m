function f = f_wdbc(w, D, mu)

    xp = [D(1:30, :); ones(1, length(D))];
    yp = D(end, :);
    
    P = length(D);
    
    f = ( (1/P) * sum(log(1 + exp((-yp).*(w' * xp)))) ) + ( (mu/2)*norm(w, 2)^2);
end