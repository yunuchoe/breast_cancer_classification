function g = g_wdbc(w_hat, D, mu)

    xp = [D(1:30, :); ones(1, length(D))];
    yp = D(end, :);
    
    P = length(D);
    
    num1 = -yp .* xp;
    num2 = exp((-yp) .* (w_hat' * xp));
    denom = 1 + exp((-yp) .* (w_hat' * xp));
    
    g = ( (1/P) * sum(((num1.*num2)./denom), 2) + (mu*w_hat) );
end