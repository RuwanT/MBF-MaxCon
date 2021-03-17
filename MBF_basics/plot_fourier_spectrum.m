function plot_fourier_spectrum(fourier_coeff, degrees, lc)

fch = zeros(1, degrees+1);
for i=0:degrees
    
    rows = fourier_coeff.degree==i;
    if sum(rows) > 0
        fc = fourier_coeff.fc(rows);
        fch(i+1) = sum(fc.^2);
    end
    
end
% fc = fc/sum(fc);
plot(0:degrees, fch, lc); hold on;
% plot(0:degrees, 1-cumsum(fc), lc); hold on;
xlabel('Degree (k)')
ylabel('$W_k = \sum_{s \subset \left [ n \right ], \left | s \right | = k } \hat{f} \left ( s \right )^2$','Interpreter','latex')