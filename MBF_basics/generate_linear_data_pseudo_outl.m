function X = generate_linear_data_pseudo_outl(N, sig, osig, d, out_frac)

X = [];
for i=1:length(N)
    m = randn(d-1, 1); 
    c = randn; 
    
    xo = rand(d-1,N(i)); 
    yo = m'*xo + repmat(c,1,N(i)); 
    
    % Corrupt data 
    x = xo + sig*randn(d-1,N(i));
    y = yo + sig*randn(1,N(i)); 
    
    % Add outliers.
    t = out_frac; 
    t = round(N(i)*t); 

    sn = sign(y(1:t) - m'*xo(:, 1:t)-c); 
    y(1:t) = y(1:t) + double(sn-~sn)*osig.*rand(1, t);
    
    x = x'; 
    y = y';
    x = [x, ones(N(i), 1)]; 
    
    X = [X; x,y];
    
end