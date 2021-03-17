function [maxcon_set, outl] = estimate_largest_feaseable_set(fourier_coeff, nbits)

% Identify the max concensus set using estimated fourier coefficients.
% Inputs:
%   fourier_coeff: fourier coefficients
%   nbits: number of input bits
%
% Output:
%   maxcon_set: maxcon index
%   outl: inx not in maxcon


maxcon_set = [];
outl = 1:nbits;
found_sol = false;
for i = nbits:-1:0
    xs = nchoosek(1:nbits,i);
    for j = 1:size(xs,1)
        inx = xs(j,:);
        x = zeros(1,nbits);
        x(inx) = 1;
        f_est = fourier_reconstruct(fourier_coeff, x, nbits);   % Estimate value from fourier coef
        f_est = int8(sign(f_est));
        if f_est == 1
            maxcon_set = inx;
            found_sol = true;
            break;
        end
    end
    if found_sol
        outl = setdiff(outl, maxcon_set);
        break;
    end
end

end