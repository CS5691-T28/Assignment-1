x = -10:0.1:10;
n = size(x,2);
n
f = exp(cos(2*3.14*x))+x;
figure;
ax1 = subplot(2,2,1)
plot(ax1,x,f);
title(ax1,'Plot of y=\ite^{cos(2\pix)}+x');
xlabel(ax1,'x');
ylabel(ax1,'y');

%Adding Gaussian Noise
r = normrnd(0,1,1,n);
t = r+f;
ax2 = subplot(2,2,2)
plot(ax2,x,t);
title(ax2,'Plot of y=\ite^{cos(2\pix)}+x with Random noise');
xlabel(ax2,'x');
ylabel(ax2,'t');

%x - input values, t- output values, Need to fit parameters
m = 3;
A = zeros(m,m);
b = zeros(1,m);
for i = 1:m
    for j = 1:m
     A(i,j) = sum(x.^(i+j-2));
    end
end 
for j = 1:m
    b(j) = sum(t.*(x.^(j-1)));
end
%A
b = b';
%b
w = linsolve(A,b);
%w
%plot_x = -10:1:10;
y = w(1)+w(2)*x+w(3)*(x.^2);
ax3 = subplot(2,2,3);
plot(x,t,'color','b'); hold on;
plot(x,y,'color','r');
title('Polynomial Fitting curve with m=3');

ax4 = subplot(2,2,4);
p = polyfit(x,t,3);
t2 = polyval(p,x);
plot(x,t,'color','b');hold on;
plot(x,t2,'color','r');
title('With standard function');
