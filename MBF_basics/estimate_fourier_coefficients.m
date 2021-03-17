function fourier_coeff = estimate_fourier_coefficients(fittingfn, nbits, method, params)

% Estimate the fourier coefficients of a function.
% Inputs:
%   fittingfn: query access to boolean function
%   nbits: number of input bits
%   method: method to sample binary vectors. options: <'full-hypercube',
%           'uniform-sampling', Goldreich-Levin, 'MBF-ODonnell-2005'>
%   params (optional): structure of parameters specific to each method
%
% Output:
%   fourier_coeff: fourier coefficients in a structure with fields 
%       fc - fourier coeff
%       parity - basis function (represented as a decimal number)
%       degree - degree of the basis

if strcmp(method,'uniform-sampling')
    
    if ~(isfield(params, 'degree'))
        error('Method specific parameters not defined.')
    elseif isfield(params, 'nsamples')
        nsamples = params.nsamples;
    elseif isfield(params, 'epsilon') && isfield(params, 'delta') && isfield(params, 'd')
        epsilon = params.epsilon;   % expected error in estimate
        delta = params.delta;       % 1 - confidence in estimate
        d = params.d;               % max significant degree 
        nsamples = floor(0.5*sqrt((nbits^d)/epsilon)*log2(2*(nbits^d)/delta)); % Low-Degree Algorithm
    else
        error('Method specific parameters not defined.')
    end
    
    degree = params.degree;
    n_degree = length(degree);
    
    % init fourier coefficent table
    fourier_coeff = init_fourier_coeffs(nbits, degree);
    
    % iterate over uniform random sampled x in [nbits]
    for nrun=1:nsamples
        
        x = randsrc(1,nbits,[0,1]);     % random sample x
       
        f_value = feval(fittingfn, x);  % evaluate function value

        for i = 1:n_degree
            rows = find(fourier_coeff.degree==degree(i));
            parity = de2bi(fourier_coeff.parity(rows), nbits);  %TODO: only work upto 52 bits
            parity = double((-1).^sum(x.*parity,2));
            fourier_coeff.fc(rows) = fourier_coeff.fc(rows) + parity*double(f_value); 
        end
    end
    fourier_coeff.fc = fourier_coeff.fc/nsamples;   % normalize

elseif strcmp(method,'uniform-influence')
    if ~(isfield(params, 'nsamples')) 
        error('Method specific parameters not defined.')
    end
    nsamples = params.nsamples;
    nsamples2 = nsamples/2;
    fixed_bits = params.nsamples;
    
    fourier_coeff.fc = [];
    fourier_coeff.parity = [];
    fourier_coeff.degree = [];
    
    X = zeros(nsamples2, nbits);
    F = zeros(nsamples2, 1);
    for i = 1:nsamples2
        x = randsrc(1,nbits,[0,1]);     % random sample x
        F(i) = feval(fittingfn, x);     % evaluate function value
        X(i,:) = x;
    end
    
    X_ = zeros(nsamples2, nbits);
    F_ = zeros(nsamples2, 1);
    for i =1:nbits
        for k = 1:nsamples2
            x = X(k,:);
            x(i) = ~x(i);
            F_(k) = feval(fittingfn, x);     % evaluate function value
            X_(k,:) = x;
        end
        fourier_coeff.fc(i) = estimate_fc_OD([X;X_], [F;F_], i);
        fourier_coeff.parity(i) = 2^(i-1);
    end
        
elseif strcmp(method,'full-hypercube')
    
    if ~(isfield(params, 'degree')) 
        error('Method specific parameters not defined.')
    end
    degree = params.degree;
    n_degree = length(degree);
    
    % init fourier coefficent table
    fourier_coeff = init_fourier_coeffs(nbits, degree);
    two2N = 2^nbits;
    
    % iterate over all possible subsets of [nbits]
    for nrun=0:two2N-1
        x = de2bi(nrun, nbits);

        f_value = feval(fittingfn, x);  % evaluate function value

        for i = 1:n_degree
            rows = find(fourier_coeff.degree==degree(i));
            parity = de2bi(fourier_coeff.parity(rows), nbits);  %TODO: only work upto 52 bits
            parity = double((-1).^sum(x.*parity,2));
            fourier_coeff.fc(rows) = fourier_coeff.fc(rows) + parity*double(f_value); 
        end
    end
    fourier_coeff.fc = fourier_coeff.fc/two2N;   % normalize
    
elseif strcmp(method,'Goldreich-Levin')
    
    if ~(isfield(params, 'nsamples') && isfield(params, 'T')) 
        error('Method specific parameters not defined.')
    end
    T2 = params.T;
    nsamples = params.nsamples;
    
    L = struct('S', [], 'k', 0, 'W', 1, 'fc', 0); %initialize list
    
    splits_exist = true;    
    while splits_exist
        [B, Binx, splits_exist] = get_bag2split(L, nbits);
        if splits_exist
            L(Binx) = [];
            [B1, B2] = split_bags_Goldreich_Levin(B);
            [B1.W, B1.fc] = estimate_bag_weight_Goldreich_Levin(B1, fittingfn, nbits, nsamples);
            [B2.W, B2.fc] = estimate_bag_weight_Goldreich_Levin(B2, fittingfn, nbits, nsamples);
            
            if B1.W > T2/2 
                L(end+1) = B1;
            end
            
            if B2.W > T2/2 
                L(end+1) = B2;
            end
        else
            % get better estimate of fourier coefficents. Why is this
            % necessary ???
            ns = nsamples*10;
            X = zeros(ns, nbits);
            F = zeros(ns, 1);
            for i = 1:ns
                x = randsrc(1,nbits,[0,1]);     % random sample x
                F(i) = feval(fittingfn, x);     % evaluate function value
                X(i,:) = x;
            end
            for i = 1:length(L)
                L(i).fc = estimate_fc_OD(X, F, L(i).S);
                % L(i).fc = estimate_fc_gl(fittingfn, L(i), nbits, nsamples*10);
            end
        end
    end
    
    % Arrange the estimated weights into fourier coefficient structure
    if length(L) < 1
        error("No significant fourier coefficents found. something wrong!")
    else
        fourier_coeff.fc = zeros(length(L),1);
        fourier_coeff.parity = zeros(length(L),1);
        fourier_coeff.degree = zeros(length(L),1);
        for i=1:length(L)
            B = L(i);
            parity = B.S;
            if length(parity) < 1
                fourier_coeff.degree(i) = 0;
                fourier_coeff.fc(i) = B.fc;
                fourier_coeff.parity(i) = 0;
            else
                fourier_coeff.degree(i) = length(parity);
                fourier_coeff.fc(i) = B.fc;
                fourier_coeff.parity(i) = sum(2.^(parity-1), 2);
            end
        end
    end
    
elseif strcmp(method,'MBF-ODonnell-2005')
    
    if ~(isfield(params, 'nsamples') && isfield(params, 'epsilon') && isfield(params, 'C')) 
        error('Method specific parameters not defined.')
    end
    
    nsamples = params.nsamples;
    epsilon = params.epsilon;
    C = params.C;
    
    fourier_coeff.degree = [];
    fourier_coeff.fc = [];
    fourier_coeff.parity = [];
    
    X = zeros(nsamples, nbits);
    F = zeros(nsamples, 1);
    for i = 1:nsamples
        x = randsrc(1,nbits,[0,1]);     % random sample x
        F(i) = feval(fittingfn, x);     % evaluate function value
        X(i,:) = x;
    end
    
    Inf_i_f = mean((X*(-2)+1).*F, 1);
    I_f = sum(Inf_i_f);
    
%     figure
%     plot(sort(Inf_i_f))
    
    t = 2*I_f/epsilon;
    
    tau = C^(-t);
    J = find(Inf_i_f > tau/2);
    if length(J) < 1
        warning('Warning: didnt pick any coordinate with influence');
    end
    
    t_ = min(length(J), t);
    nn = 1:nbits;
    
    fourier_coeff.degree = [fourier_coeff.degree; 0];
    fourier_coeff.fc = [fourier_coeff.fc; mean(F)];
    fourier_coeff.parity = [fourier_coeff.parity; 0];
    for i = 1:length(J)
        fourier_coeff.degree = [fourier_coeff.degree; 1];
        fourier_coeff.fc = [fourier_coeff.fc;Inf_i_f(J(i))];
        fourier_coeff.parity = [fourier_coeff.parity; sum(2.^(J(i)-1), 2)];
    end
    
    if length(J) > 1
        for i = 2:t_
            J_j = nchoosek(J, i);
            for j = 1:size(J_j,1)
                S = nn(J_j(j,:));
                fc = estimate_fc_OD(X, F, S);
                fourier_coeff.degree = [fourier_coeff.degree; length(S)];
                fourier_coeff.fc = [fourier_coeff.fc;fc];
                fourier_coeff.parity = [fourier_coeff.parity; sum(2.^(S-1), 2)];
            end
        end
    end
    
        
else
    error("Incorrect sampling method. try <random_sampling, full_hypercube>")
end

end


function fourier_coeff = init_fourier_coeffs(nbits, degree)

    n_degree = length(degree);
    
    if n_degree < 1
        error("Need to have atleast one element in degrees")
    end
    
    parity = nchoosek(1:nbits,degree(1));   % TODO: may not be efficient
    nelements = size(parity,1);
    fc = zeros(nelements,1);
    deg = ones(nelements,1) * degree(1);
    inx = sum(2.^(parity-1), 2);            % 'right-msb' aligned with default de2bi

    for i = 2:n_degree
        parity = nchoosek(1:nbits,degree(i));   % TODO: may not be efficient
        nelements = size(parity,1);
        fc = [fc; zeros(nelements,1)];
        deg = [deg; ones(nelements,1) * degree(i)];
        inx = [inx; sum(2.^(parity-1), 2)];
    end
    
    fourier_coeff.fc = fc;
    fourier_coeff.parity = inx;
    fourier_coeff.degree = deg;
    
end


function [B1, B2] = split_bags_Goldreich_Levin(B)
    B1.k = B.k + 1;
    B2.k = B.k + 1;
    
    B1.S = B.S;
    B2.S = [B.S, B.k+1];
    
    B1.W = 0;
    B2.W = 0;
    
    B1.fc = 0;
    B2.fc = 0;
end

function [W, fc] = estimate_bag_weight_Goldreich_Levin(B, fittingfn, nbits, nsamples)
    W = 0;
    fc = 0;
   
    for i = 1:nsamples
        
        % TODO: Not sure about the bits that are not in S or k+1-n
        % Maybe set them to 0 (not done yet)
        z_y1 = randsrc(1, nbits, [0,1]);     % random sample x
        y2 = randsrc(1, length(B.S), [0,1]);

        if isempty(B.S)
            z_y2 = z_y1;
            p_1 = 1.0;
            p_2 = 1.0;
        else
            z_y2 = z_y1;
            z_y2(B.S) = y2;
            p_1 = (-1)^sum(z_y1(B.S));
            p_2 = (-1)^sum(z_y2(B.S));
        end
        f_1 = double(feval(fittingfn, z_y1));
        f_2 = double(feval(fittingfn, z_y2));

        W = W + f_1*p_1*f_2*p_2;
        fc = fc + f_1*p_1;

    end
    
    W = W/nsamples;
    fc = fc/nsamples;
    
end


function [B, Binx, splits_exist] = get_bag2split(L, nbits)

    % TODO: Currently doing breadth first search. Try others. 
    Binx = 1;
    kd_card = nbits;
    for i = 1:length(L)
        if L(i).k < kd_card
            kd_card = L(i).k;
            Binx = i;
        end
    end
    
    B = L(Binx);
    splits_exist = kd_card ~= nbits;
end


% function fc = estimate_fc_gl(fittingfn, B, nbits, nsamples)
% 
%     W_fc = 0;
%     for i = 1:nsamples
%         z_y1 = randsrc(1, nbits, [0,1]);     % random sample x
%         f_1 = feval(fittingfn, z_y1);
%         if isempty(B.S)
%             parity = zeros(1, nbits);  %TODO: only work upto 52 bits
%         else
%             parity = de2bi(sum(2.^(B.S-1), 2), nbits);  %TODO: only work upto 52 bits
%         end
%         p_1 = (-1).^sum(z_y1.*parity,2);
% 
%         W_fc = W_fc + double(f_1)*double(p_1);
%     end
%     fc = W_fc/nsamples;
%     
% end


function fc = estimate_fc_OD(X, F, S)

    [nsamples, ~] = size(X);
    if isempty(S)
        parity = ones(nsamples, 1);
    elseif length(S) == 1
        parity = (-1).^X(:, S);
    else
        parity = (-1).^sum(X(:, S), 2);
    end
    
    fc = mean(F.*parity);
    
end