% ECE 403 - Lab 3

% 1. From the course website download data matrix D_wdbc.mat.
%load("D_wdbc.mat")


% 2. Follow Sec. 3.1 above to generate normalized training and test data sets and their labels.
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


% 3. Based on (E3.2) and (E3.3), prepare two MATLAB functions for evaluating the regularized
% cost function and its gradient, respectivel

% prepared E3.2 in f_wdbc.m and E3.3 in g_wdbc.m - same equation structure
% as in assignment 4


% 4. Prepare MATLAB code to minimize E_LR(w_hat) in (E3.2) using the GD algorithm. The remarks
% below are intended to assist to get this done smoothl
D = [Xtr; ytr];


% 5. Set initial point w_hat = 0 and run the code prepared in Item 4.4 to obtain parameters {w*, b*}
% for the following settings:
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


% 6. Use the four solutions {w*, b* }
% obtained from 5. above to specify the classifier in
% (E3.4) and apply it to the 284 test samples prepared in Item 4.2 above. 
% Report the confusion matrix for each case, and comment on the results obtained

% encapsualted label and confusion matrix generation into sperate function

fprintf("(i) \n")
confusion_matrix(ws1, Xte, yte)

fprintf("\n(ii) \n")
confusion_matrix(ws2, Xte, yte)

fprintf("\n(iii) \n")
confusion_matrix(ws3, Xte, yte)

fprintf("\n(iv) \n")
confusion_matrix(ws4, Xte, yte)