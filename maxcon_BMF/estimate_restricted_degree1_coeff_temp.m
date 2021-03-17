function [fc, f_evals, X_, F_] = estimate_restricted_degree1_coeff(fittingfn, X_, F_, nbits, restricted_bits, restriction, nsamples, cprob, S)
% Estimate the influence of each bit in S
% Inputs:
%   fittingfn - 
%   nbits - 
%   restricted_bits - index of bits restricted to 0
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
    
    fc = zeros(1,length(S));
    X = zeros(nsamples, nbits);
    F = zeros(nsamples, 1);
    
    % if ~isempty(restricted_bits)
    %     X(:,restricted_bits ) = repmat(restriction,nsamples,1 );
    % end
    
    non_restricted_bits = setdiff(1:nbits, restricted_bits);
    
    Xelements = length(F_);
    
    inx_ord = randperm(Xelements);
    
    % Generate half of the samples by using previous
    j=1;
    for i = 1:2:nsamples
        F(i) = F_(inx_ord(j));     % evaluate function value
        X(i,:) = X_(inx_ord(j),:);
        j = j+1;
    end
    
    % generate the other half by flipping relevent bits
    for j = 1:length(S)
        tbit = S(j);
        if (sum(restricted_bits==tbit) < 1) 
            for i = 1:2:nsamples
                X(i+1, :) = X(i, :);
                X(i+1, tbit) = ~X(i+1, tbit);
                
                if F(i) == -1 && X(i, tbit) == 0
                    F(i+1) = -1;
%                     X_(Xelements+1, : ) = X(i+1, :);
%                     F_(Xelements+1) = F(i+1);
%                     Xelements = Xelements + 1;
                elseif F(i) == 1 && X(i, tbit) == 1
                    F(i+1) = 1;
%                     X_(Xelements+1, : ) = X(i+1, :);
%                     F_(Xelements+1) = F(i+1);
%                     Xelements = Xelements + 1;
                else
                    F(i+1) = feval(fittingfn, X(i+1, :));     % evaluate function value
                    f_evals = f_evals + 1;
%                     X_(Xelements+1, : ) = X(i+1, :);
%                     F_(Xelements+1) = F(i+1);
%                     Xelements = Xelements + 1;
                end
                
            end
            fc(j) = estimate_fc_OD(X, F, tbit);
        else
            error('interesting - look in estimate_restricted...')
        end
    end

end

function fc = estimate_fc_OD(X, F, S)

    [nsamples, ~] = size(X);
    if isempty(S)
        parity = ones(nsamples, 1);
    elseif length(S) == 1
        parity = (-1).^X(:, S);
    else
        parity = (-1).^sum(X(:, S), 2);
    end
    
    fc = 1* mean(F.*parity);
    
end