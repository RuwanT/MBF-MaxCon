function rhd = relative_hamming_distance(fourier_coeff, fittingfn, nbits, method, nsamples)

% Estimate the relative hamming distance between function described by
% fittingfn/th and its fourier reconstruction using fourier_coeff.
% Inputs:
%   fourier_coeff: fourier coefficients
%   fittingfn: query access to boolean function
%   nbits: number of input bits
%   method: method to sample binary vectors
%   nsamples: number of samples for sampling based methods
%
% Output:
%   rhd: relative hamming distance


if strcmp(method,'uniform_sampling')
    rhd = 0;
    for nrun=1:nsamples
        x = randsrc(1,nbits,[0,1]);     % random sample x
        
        f_value = feval(fittingfn, x);  % Evaluate function value
        f_est = fourier_reconstruct(fourier_coeff, x, nbits);   % Estimate value from fourier coef
        f_est = int8(sign(f_est));
        
        if f_est ~= f_value
            rhd = rhd + 1;
        end
    end
    rhd = rhd/nsamples;

elseif strcmp(method,'full_hypercube')
    rhd = 0;
    two2N = 2^nbits;
    for nrun=0:two2N-1
        x = de2bi(nrun, nbits);
        
        f_value = feval(fittingfn, x);  % Evaluate function value
        f_est = fourier_reconstruct(fourier_coeff, x, nbits);   % Estimate value from fourier coef
        f_est = int8(sign(f_est));
        
        if f_est ~= f_value
            rhd = rhd + 1;
        end
    end
    rhd = rhd/two2N;
        
else
    error("Incorrect sampling method. try <random_sampling, >")
end
