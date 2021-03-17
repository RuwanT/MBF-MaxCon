function pl = plot_maxcon_hypo(theta, th, lt)
x = [0,1;1,1];
y = x*theta;

pl = plot(x(:,1), y+th, lt);
plot(x(:,1), y-th, lt)
end