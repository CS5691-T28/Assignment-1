x = 0:0.01:1;
N = size(x,2);
t = exp(cos(2*3.14*x))+x;
r = normrnd(0,0.5,1,N);
t = t + r;
%N
rand_perm = randperm(N);
%rand_perm
train_ind = sort(rand_perm(1:70));
val_ind = sort(rand_perm(71:80));
test_ind = sort(rand_perm(81:end));

x_train = x(train_ind);
t_train = t(train_ind);
x_val = x(val_ind);
t_val = t(val_ind);
x_test = x(test_ind);
t_test = t(test_ind);
n = size(x_train,2);

plot(x,exp(cos(2*3.14*x))+x);hold on;
scatter(x_train,t_train);
title('Plot of y=\ite^{cos(2\pix)}+x for N=70');
xlabel('x');
ylabel('y');
hold off;

%x_train

%x - input values, t- output values, Need to fit parameters by least
%sqaured error method.
M = [0,1,3,9,10,15];
etrain = zeros(1,size(M,2));
etest = zeros(1,size(M,2));

for index = 1:size(M,2)
    m = M(index);
    [erms_train, erms_test] = fitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, m);
    etrain(index) = erms_train;
    etest(index) = erms_test;
    
end   
figure;
plot(M,etrain); hold on;
xlabel('M');
plot(M, etest); hold off;

% ax4 = subplot(2,2,4);
% p = polyfit(x_train,t_train,m);
% t2 = polyval(p,x_train);
% plot(x_train,t2,'color','b');hold on;
% plot(x,t,'color','r');
% title('With standard function');




function e_rms = calcerror(x,y,t,n)
 e_rms = (sum((t-y).^2)/n).^0.5;
end

function [erms_train,erms_test] = fitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, m)
 A = zeros(m+1,m+1);
 b = zeros(1,m+1);
 n = size(x_train,2);
 phi = zeros(m+1,n);
 for i = 1:m+1
     for j = 1:m+1
      A(i,j) = sum(x_train.^(i+j-2));
     end
 end 
 for j = 1:m+1
     b(j) = sum(t_train.*(x_train.^(j-1)));
 end
 %A
 b = b';
 %b
 w = linsolve(A,b);
 for i=1:m+1
     phi(i,:) = x_train.^(i-1);
 end 
 y_train = w'*phi;
 figure;
 plot(x_train,y_train,'color','b'); hold on;
 scatter(x_train,t_train);
 title(['Polynomial Fitting curve with m=' num2str(m) '']);
 
%For Test data set
n_test = size(x_test,2);
phi_test = zeros(m+1, n_test);
for i=1:m+1
    phi_test(i,:) = x_test.^(i-1);
end  
y_test = w'*phi_test;

print('RMS Error for training set');
erms_train = calcerror(x_train, y_train, t_train,n);
%e_train_stdfunc = calcerror(x_train, t2, t_train, n);
erms_test = calcerror(x_test, y_test, t_test, n_test);

erms_train
erms_test
 
end
