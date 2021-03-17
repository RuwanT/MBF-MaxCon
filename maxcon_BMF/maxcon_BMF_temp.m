function [fixed_bits, f_evals] = maxcon_BMF(method, fittingfn, nbits, minmaxfn, nsamples, q, use_local)
    % fittingfn : Quary access to BMF
    % nbits: Number of input bits
    % minmaxfn: L-inf function access
    % nsamples: Number of samples to calculate fourier coeff
    % q: Positive Probability P(x_1 = 1)

    all_bits = 1:nbits;
    fixed_bits = [];
    
    f_evals = 0;
    
    X = zeros(nsamples, nbits);
    F = zeros(nsamples, 1);
    
    % Generate the samples by random sampling
    for i = 1:1:nsamples
        x = randsrc(1, nbits, [0, 1; 1-q, q]);     % random sample x
        F(i) = feval(fittingfn, x);     % evaluate function value
        X(i,:) = x;
    end
    f_evals = nsamples;

    for i = 1:nbits

        remaining_bits = setdiff(all_bits, fixed_bits);

        if strcmp(method, 'maxcon-Linf')
            % Get the L-infinity fit to remaining data
            sol = feval(minmaxfn, remaining_bits);
            basis = sol.basis;
        else
            basis = 1:length(remaining_bits);
        end

        % Estimate deg-1 coeficents for basis datapoints of the restricted
        % function
        [fc, fevals, X, F] = estimate_restricted_degree1_coeff(fittingfn, X, F, nbits, fixed_bits, zeros(1, length(fixed_bits)), nsamples, q, remaining_bits(basis));
        f_evals = f_evals + fevals;
        % Remove the basis point with max influence from the dataset 
        [~, rbit] = max(fc);
        rbit = basis(rbit);
        rbit = remaining_bits(rbit);
        fixed_bits = [fixed_bits, rbit];

        % check for feasiability
        x = ones(1, nbits);
        x(fixed_bits) = 0;
        f_value = feval(fittingfn, x);  % evaluate function value
        if f_value == 1 % feaseable
            break
        end
        
        remove_inx = X(:, rbit)==1;
        
        F(remove_inx) = [];
        X(remove_inx,:) = [];
        
        remove_inx = sum(X, 2)<8+1;
        F(remove_inx) = [];
        X(remove_inx,:) = [];
        
        [X,ia,~] = unique(X,'rows');
        F = F(ia);
        
        missing = nsamples/2 - length(F);
        non_restricted_bits = setdiff(1:nbits, fixed_bits);
        
        if missing>0
            for j = length(F)+1:nsamples/2
                x = zeros(1,nbits);
                x(non_restricted_bits) = randsrc(1, nbits-length(fixed_bits), [0, 1; 1-q, q]);
                F(j) = feval(fittingfn, x);     % evaluate function value
                X(j,:) = x;
                f_evals = f_evals + 1;
            end
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