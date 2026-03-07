load("D_wdbc.mat")

% get data sets
Dtr = D_wdbc(:,1:285);
Dte = D_wdbc(:,286:569);

% normalize training data
Xtr = zeros(30,285);
m = zeros(1,30);
v = zeros(1,30);
for i = 1:30
    xi = Dtr(i,:);
    m(i) = mean(xi);
    v(i) = sqrt(var(xi));
    Xtr(i,:) = (xi - m(i))/v(i);
end

% normalize test data
Xte = zeros(30,284);
for i = 1:30
    xi = Dte(i,:);
    Xte(i,:) = (xi - m(i))/v(i);
end

% assign labels
ytr = Dtr(31,:);
yte = Dte(31,:);

D = [Xtr; ytr];
w_hat = zeros(31,1);

% (i) mu = 0 and k = 10
mu = 0;
k = 10;

[ws1,fs1,k1] = grad_desc('f_wdbc', 'g_wdbc', w_hat, D, mu, k);

% (ii) mu = 0.1 and k = 10
mu = 0.1;
k = 10;

[ws2,fs2,k2] = grad_desc('f_wdbc', 'g_wdbc', w_hat, D, mu, k);

% (iii) mu = 0 and k = 30
mu = 0;
k = 30;

[ws3,fs3,k3] = grad_desc('f_wdbc', 'g_wdbc', w_hat, D, mu, k);

% (iv) mu = 0.075 and k = 30
mu = 0.075;
k = 30;

[ws4,fs4,k4] = grad_desc('f_wdbc', 'g_wdbc', w_hat, D, mu, k);



fprintf("(i) \n")
confusion_matrix(ws1, Xte, yte)

fprintf("\n(ii) \n")
confusion_matrix(ws2, Xte, yte)

fprintf("\n(iii) \n")
confusion_matrix(ws3, Xte, yte)

fprintf("\n(iv) \n")
confusion_matrix(ws4, Xte, yte)
