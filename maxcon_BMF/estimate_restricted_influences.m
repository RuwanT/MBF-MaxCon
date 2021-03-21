function [fc, f_evals] = estimate_restricted_influences(fittingfn, nbits, restricted_bits, restriction, nsamples, cprob, S)
% Estimate the influence of each bit in S
% Inputs:
%   fittingfn - Quary access to BMF
%   nbits - Number of input bits
%   restricted_bits - index of bits restricted (or fixed)
%   restriction - vector of values for restricted bits
%   nsamples - number of function queries to be made (need to be even number)
%   cprob - prob of sampling a 1 in generating bit vectors
%   S - set of bits for which the degree-1 fourier coefs are calculated
%
% Outputs:
%   fc: Fourier coefficents
%   f_evals - Number of function ecaluation

    if ~exist('cprob','var')
        cprob = 0.5;
    end
    if ~exist('S','var')
        S = 1:nbits;
    end
    
    f_evals = 0;
    
    non_restricted_bits = setdiff(1:nbits, restricted_bits);
    
    %%% Use degree 1 fourier equations to compute influence 
    %{
    Inf = zeros(1,length(S));
    % Generate half of the samples by random sampling
    for i = 1:1:nsamples
        x = zeros(1,nbits);
        x(non_restricted_bits) = randsrc(1, nbits-length(restricted_bits), [0, 1; 1-cprob, cprob]);     % random sample x
        Fi = double(feval(fittingfn, x));     % evaluate function value
        f_evals = f_evals + 1;
        
        Inf = Inf + (1/nsamples/2)*Fi*((-1).^x(S));
        for j = 1:length(S)
            tbit = S(j);
            if Fi == -1 && x(tbit) == 0
                Inf(j) = Inf(j) + (1/nsamples/2)*Fi*(-1)^(~x(tbit));
            elseif Fi == 1 && x(tbit) == 1
                Inf(j) = Inf(j) + (1/nsamples/2)*Fi*(-1)^(~x(tbit));
            else
                x_hat = x;
                x_hat(tbit) = ~x_hat(tbit);
                Fi_hat = double(feval(fittingfn, x_hat));     % evaluate function value
                f_evals = f_evals + 1;
                Inf(j) = Inf(j) + (1/nsamples/2)*Fi_hat*(-1)^(x_hat(tbit));
                %if Fi_hat ~= Fi
                %    Inf(i,j) = 1;
                %end
            end
                
            
        end
        
    end
    
    %fc = mean(Inf);
    fc = Inf;
    %}
    
    %%% Use definition of influence equations to compute influence 
    % {
    Inf = zeros(nsamples,length(S));
    for i = 1:1:nsamples
        x = zeros(1,nbits);
        x(non_restricted_bits) = randsrc(1, nbits-length(restricted_bits), [0, 1; 1-cprob, cprob]);     % random sample x
        Fi = feval(fittingfn, x);     % evaluate function value
        f_evals = f_evals + 1;

        for j = 1:length(S)
            tbit = S(j);
            if Fi == -1 && x(tbit) == 0
                Inf(i,j) = 0;
            elseif Fi == 1 && x(tbit) == 1
                Inf(i,j) = 0;
            else
                x_hat = x;
                x_hat(tbit) = ~x_hat(tbit);
                Fi_hat = feval(fittingfn, x_hat);     % evaluate function value
                f_evals = f_evals + 1;
                
                if Fi_hat == Fi
                   Inf(i,j) = 0;
                else
                   Inf(i,j) = 1;
                end
            end
                
            
        end
        
    end
    
    fc = mean(Inf);
    
    %}
end

