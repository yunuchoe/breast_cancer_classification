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

% let mu = 0 and k = 10
mu = 0;
k = 10;

[ws,fs,k] = grad_desc('f_wdbc', 'g_wdbc', w_hat, D, mu, k);

% results
fprintf("Confusion matrix: \n")
confusion_matrix(ws, Xte, yte)
