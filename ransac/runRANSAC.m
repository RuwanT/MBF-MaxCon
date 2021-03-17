function [bestTheta, bestInliers, trialcount] = runRANSAC(data, th, nbits, p, maxIter, autoStop, finalUpdate, maxrunTime)
% New folder
% Inputs:
%   data: [d x n] matrix of data with [x;y]
%   th: Inlier threshold
%   nbits: number of data points
%   p: minimal sample set cardinality
%   maxIter: Max Number of function queries 
%   Autostop: Boolean indicating if ransac stopping criterion to be used
%   finalUpdate: Final update of ransac with all inliers so far
%   maxrunTime: Maximum runtine (unused if 0)


    inmethodtic = tic;

    fittingfn = @myFitTchebycheff;
    distfn    = @lineptdist;
    degenfn   = @isdegenerate;
    
    rp = 0.99;         % Desired probability of choosing at least one sample

    if maxrunTime ~= 0
        maxIter = inf;
    end
    
    calcIter = maxIter;
    trialcount = 0;
    
    
    bestScore = 0;
    bestTheta = [];
    bestInliers = [];
    
    
    while (trialcount < min(maxIter, calcIter))
        sample_inx = randperm(nbits);
        sample_inx = sample_inx(1:p);
        
        theta = feval(fittingfn, data(:, sample_inx));
        if isempty(theta)
                degenerate = 1;
                continue;
        end
        
        [inliers, theta] = feval(distfn, theta, data, th);
        trialcount = trialcount + 1;
    
        % Find the number of inliers to this model.
        ninliers = length(inliers);
        if ninliers > bestScore    % Largest set of inliers so far...
            bestScore = ninliers;  % Record data for this model
            bestInliers = inliers;
            bestTheta = theta;
        end
        
        if autoStop==1
            % Update estimate of N, the number of trials to ensure we pick,
            % with probability p, a data set with no outliers.
            fracinliers =  bestScore/nbits;
            pNoOutliers = 1 -  fracinliers^p;
            pNoOutliers = max(eps, pNoOutliers);  % Avoid division by -Inf
            pNoOutliers = min(1-eps, pNoOutliers);% Avoid division by 0.
            calcIter = log(1-rp)/log(pNoOutliers);
        end
        
        ttime = toc(inmethodtic);
        if (maxrunTime < ttime) && (maxrunTime ~= 0)
            break
        end
        
        
        
    end
    
    if finalUpdate==1
        %perform LS on the final consensus set
        theta = feval(fittingfn, data(:, bestInliers));
        [inliers, theta] = feval(distfn, theta, data, th);
        if (numel(bestInliers)<numel(inliers))
            bestInliers = inliers;
            bestTheta = theta;
        end
    end

end


function [inliers, theta] = lineptdist(theta, data, th)

    X = data(1:end-1, :)';  
    Y = data(end, :)'; 
    d = X*theta - Y; 
    
    inliers = find(abs(d) < th);  
    
    if numel(inliers) > size(data,1) + 2
        dst = myFitTchebycheff_dist(data(:,inliers));
        if dst > th+eps
            error("something went wrong")
        end
    end
end


function r = isdegenerate(X)
    r = 0; 
end