function d = fit_linear_dist(data)

    X = data(1:end-1, :)';
    
    Y = data(end, :)';
    X = [X, ones(length(Y),1)];
    
    theta = (X'*X)\(X'*Y);
    
    d = X*theta - Y; 
    
    d = max(abs(d));  

end