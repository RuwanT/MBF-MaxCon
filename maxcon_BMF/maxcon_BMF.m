%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consensus Maximisation Using Influences of Monotone Boolean Functions
% Proposed in
% Tennakoon, Ruwan and Suter, David and Zhang, Erchuan and Chin, Tat-Jun 
% and Bab-Hadiashar, Alireza "Consensus Maximisation Using Influences of 
% Monotone Boolean Functions"
% In Proceedings of the IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2021.
% 
% Copyright (c) 2021 Ruwan Tennakoon (ruwan.tennakoon@rmit.edu.au)
% School of Computing Technologies, RMIT University, Australia
% https://ruwant.github.io/
% Please acknowledge the authors by citing the above paper in any academic 
% publications that have made use of this package or part of it.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fixed_bits, f_evals] = maxcon_BMF(method, fittingfn, nbits, minmaxfn, nsamples, q, use_local)
    % fittingfn : Quary access to BMF
    % nbits: Number of input bits
    % minmaxfn: L-inf function access
    % nsamples: Number of samples to calculate fourier coeff
    % q: Positive Probability P(x_1 = 1)
    % use_local: use the local expantion step if = 1

    all_bits = 1:nbits;
    fixed_bits = [];
    
    f_evals = 0;

    for i = 1:nbits

        remaining_bits = setdiff(all_bits, fixed_bits);

        if strcmp(method, 'maxcon-Linf')
            % Get the L-infinity fit to remaining data
            sol = feval(minmaxfn, remaining_bits);
            basis = sol.basis;
            basis = remaining_bits(basis);
        else
            basis = 1:length(remaining_bits);
            basis = remaining_bits(basis);
        end

        % Estimate deg-1 coeficents for basis datapoints of the restricted
        % function
        [fc, fevals] = estimate_restricted_degree1_coeff(fittingfn, nbits, fixed_bits, zeros(1, length(fixed_bits)), nsamples, q, basis);
        % [fc, fevals] = estimate_restricted_influences(fittingfn, nbits, fixed_bits, zeros(1, length(fixed_bits)), nsamples, q, basis);
        % fc = zeros(size(basis));
        % fevals = nsamples;
        
        
        f_evals = f_evals + fevals;
        % Remove the basis point with max influence from the dataset 
        [~, rbit] = max(fc);
        rbit = basis(rbit);
        fixed_bits = [fixed_bits, rbit];

        % check for feasiability
        x = ones(1, nbits);
        x(fixed_bits) = 0;
        f_value = feval(fittingfn, x);  % evaluate function value
        if f_value == 1 % feaseable
            break
        end

    end

    % perform loca expantion
    if use_local
        remaining_bits = setdiff(all_bits, fixed_bits);
        x = zeros(1, nbits);
        x(remaining_bits) = 1;

        is_sol = 1;
        i_hold = length(remaining_bits);
        while is_sol==1
            [is_sol, x, i_hold, fevals] = loca_expand(fittingfn, x, [remaining_bits, fixed_bits], i_hold);
            f_evals = f_evals + fevals;
        end
        
        fixed_bits = find(x==0);

    end

end


function [is_sol, alpha_n, i_hold, f_evals] = loca_expand(fittingfn, alpha_0, all_bits, i_hold)
    f_evals = 0;
    is_sol = 0;
    for i = i_hold+1:length(all_bits)
        alpha_n = alpha_0;
        alpha_n(all_bits(i)) = 1;
        f_value = feval(fittingfn, alpha_n);
        f_evals = f_evals + 1;
        if f_value == 1    %fesiable
            is_sol = 1;
            i_hold = i;
            return;
        end
    end
    alpha_n = alpha_0;
end