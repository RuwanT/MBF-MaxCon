function [bestThetalo, bestInlierslo, trialcount] = runLoRANSAC(data, th, nbits, p, maxIter, autoStop, finalUpdate, maxrunTime)

% Inputs:
%   data: [d x n] matrix of data with [x;y]
%   th: Inlier threshold
%   nbits: number of data points
%   p: minimal sample set cardinality
%   maxIter: Max Number of function queries 
%   Autostop: Boolean indicating if ransac stopping criterion to be used
%   finalUpdate: Final update of ransac with all inliers so far (not used)
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
    
    
    % iter = 0; % RANSAC iterations
    bestScore = 0;
    bestScorelo = 0;
    
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
            
            [loTheta, loUpdatedInliers, trialcount_] = runLocalOptimization(distfn, fittingfn, data, th, bestInliers);
            trialcount = trialcount + trialcount_;
            if numel(loUpdatedInliers)> bestScorelo
                bestScorelo = numel(loUpdatedInliers);
                bestThetalo = loTheta;
                bestInlierslo = loUpdatedInliers;
            end
            
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
    

end


function [bestTheta, bestInliers, trialcount_] = runLocalOptimization(distfn, fittingfn, data, th, minliers)

    trialcount_ = 0;
    
    loRansacMaxIter = 10;
    irlsSteps = 10;
    th_multiplier = 3; 
    
    d = size(data,1)-1;
    
    loTheta = feval(fittingfn, data(:,minliers));
    [baseinliers, loTheta] = feval(distfn, loTheta, data, th);
    trialcount_ = trialcount_ + 1;
    
    loIter = 0;
    
    bestscore = numel(baseinliers);
    bestTheta = loTheta;
    bestInliers = baseinliers;
    
    while (loIter < loRansacMaxIter && (~isempty(baseinliers)) )
        
        % Start sampling from the InlierSet (higher than
        % minimal subsets)
        loind = randsample(baseinliers, length(baseinliers)/2 );
        % Re-estimate the model
        % disp([length(loind), length(baseinliers)])
        loTheta = feval(fittingfn, data(:,loind));
        trialcount_ = trialcount_ + 1;
        [loUpdatedInliers, loTheta] = feval(distfn, loTheta, data, th);  
        
        if numel(loUpdatedInliers) > bestscore
            bestscore = numel(loUpdatedInliers);
            bestTheta = loTheta;
            bestInliers = loUpdatedInliers;
        end
        
        %Peform iteratively reweighted least square
        th_step_size = (th_multiplier*th - th)./irlsSteps;
        for loirls = 1:irlsSteps
            [loInls, loTheta] = feval(distfn, loTheta, data, th*th_multiplier - loirls*th_step_size);
            loX = data(1:d,loInls)'; loY = data(end, loInls)';
            loRes = abs(loX*loTheta - loY);
            trialcount_ = trialcount_ + 1;
            loWeight = 1./loRes; W = diag(loWeight);
            %Weighted LS:
            loTheta = (loX'*W*loX)\(loX'*W*loY);
        end
        [loUpdatedInliers, loTheta] = feval(distfn, loTheta, data, th);  
        
        if numel(loUpdatedInliers) > bestscore
            bestscore = numel(loUpdatedInliers);
            bestTheta = loTheta;
            bestInliers = loUpdatedInliers;
        end
        
        
        loIter = loIter + 1;
    end

end


function [inliers, theta] = lineptdist(theta, data, th)

    X = data(1:end-1, :)';  
    Y = data(end, :)'; 
    d = X*theta - Y; 
    
    inliers = find(abs(d) < th);  
    
%     if numel(inliers) > 10
%         d = myFitTchebycheff_dist(data(:,inliers));
%         if d > th+eps
%             error("something went wrong")
%         end
%     end
end


function r = isdegenerate(X)
    r = 0; 
end