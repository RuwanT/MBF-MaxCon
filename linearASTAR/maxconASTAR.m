%##########################################################################
%% G L O B A L L Y   O P T I M A L   C O N S E N S U S   M A X I M I S A T I O N
%% This package contains the source code which implements optimal Consensus 
% Maximisation proposed in
% T.J. Chin, P. Purkait, A. Eriksson and D. Suter
% Efficient Globally Optimal Consensus Maximisation with Tree Search, 
% In Proceedings of the IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), June 2015, Boston
% 
% Copyright (c) 2015 Pulak Purkait (pulak.purkait@adelaide.edu.au.)
% School of Computer Science, The University of Adelaide, Australia
% The Australian Center for Visual Technologies
% http://www.cs.adelaide.edu.au/directory/pulak.purkait
%% Please acknowledge the authors by citing the above paper in any academic 
%  publications that have made use of this package or part of it.
%##########################################################################

function [pk,wk,vk,nnum,xnum] = maxconASTAR(X,y, x0, th)
% BFS with Matousek for maximum consensus.

N = size(X,1);
xnum = 1; 
% x0 = rand(2, 1); 
[p, w, b] = myFitTchebycheff(X, y, x0); 
node.p = p;                     % Fitted parameters.
node.b = b; 
node.w = w;                     % Chebyshev fit criterion value.
node.v = [];                    % Violation set.
[node.h, w, p, ~, ~, xnum] = heuristic(X,y, [1:N], [], p, th, xnum);     % Heuristic value.(X,y, H, v, x0, cyc, th)
[U_cnt, v] = compute_upper(X, y, p, th); 

node.f = length(node.v)+node.h; % Evaluation value.
nnum = 0;
if (node.w<=th) 
    pk = node.p;
    wk = node.w;
    vk = node.v; 
    return;
end
if  node.h == U_cnt
    pk = p;
    wk = w;
    vk = v; 
    return;
end


node.solved = 0; 
minOUT = U_cnt; 
q = node; 

priof = node.f;
priow = node.w; 
% List of checked bases.
dictS = java.util.Hashtable; 

while 1

    [mn,~] = min(priof, [], 2);
    id = find(priof == mn); 
    [~, mp] = min(priow(id));
    mp = id(mp); 
 
    node = q(mp);
    if(node.solved)
        pk = node.p;
        wk = node.w;
        vk = node.v; 
        return; 
    end
    q(mp) = [];
    
    % disp([numel(node.v), mn, minOUT]); 
    priof(mp) = []; 
    priow(mp) = []; 
    nnum = nnum+1;
            
    % Generate child nodes.
    for i=1:length(node.b)
        xnum = xnum+1;

        child.v = sort([node.v node.b(i)]);
        entry = dictS.get(sprintf('%4ld', child.v)); 
        if numel(entry)
            continue; 
        end
        
        H = 1:N;
        H(child.v) = []; 
        [child.p, child.w,bs] = myFitTchebycheff(X(H,:),y(H), node.p); 
        child.b = H(bs);
       
        [child.h, w, p, ~, ~,xnum] = heuristic(X,y, H, [], child.p, th,xnum);   
        child.h = max(child.h, node.h-1); 
        child.f = numel(child.v)+child.h;

        dictS.put(sprintf('%4ld', child.v), 1); 
        [UN_cnt, v] = compute_upper(X(H,:), y(H), p, th); 
        if  minOUT > UN_cnt + numel(child.v)
            minOUT = UN_cnt + numel(child.v); 
        end
%         disp(minOUT); 
        if child.h == UN_cnt && child.f <= minOUT
            % Reached a to the solution; 
            child.solved = 1; 
            child.p = p; 
            child.w = w; 
            child.v = sort([child.v, H(v)]); 
%         end
        else
            child.solved = 0; 
            if child.w<=th
                % Reached a goal state.

                pk = node.p;
                wk = node.w;
                vk = child.v; 
                return;
            end
        end
        
        idx = priof > minOUT; 
        q(idx) = []; 
        priof(idx) = []; 
        priow(idx) = []; 

        q = [ q, child ]; 
        priof = [priof, child.f]; 
        priow = [priow, child.w];  
        
    end
end

end

function [ cyc, w, x0_f, chsub_old, sub_f, xnum] = heuristic(X,y, H, chsub_old, x0, th, xnum)
%       This function computes the admissible heuristics for ASTAR

% Inputs:
%   X - Data X
%   y - Data y
%   H - Remaining data indexes
%   chsub_old - removed basis last round
%   x0 - init theta
%   th - threshold
%   xnum - ?

% OUTPUTS:
%   cyc - huristic value
%   w - Chebyshev fit criterion value (distance).
%   x0_f - fitted parameters
%   chsub_old - ~
%   sub_f - ~
%   xnum - looks like a count of l-inf calculations.

[x0, val,chsub] = myFitTchebycheff(X(H,:),y(H), x0); 
xnum = xnum + 1; 

r = X(chsub_old, :)*x0 - y(chsub_old, :); 
M = find(abs(r) < val);
H = [H, chsub_old(M)]; 
chsub_old(M) = []; 

if (val>th)
    % Feasible solution not found.
    H1 = H; 
    H1(chsub) = [];
    [ cyc, w, x0_f, chsub_new, sub_f, xnum] = heuristic(X,y, H1, H(chsub), x0, th, xnum); 
else
    cyc = 0;
    x0_f = x0; 
    w = val; 
    sub_f = H; 
    return; 
end

for i=1:numel(chsub_new)
    sub_f = [sub_f, chsub_new(i)]; 
    if numel(sub_f) <= numel(chsub)
          cyc = cyc + 1;
          return; 
    end
    r = X(chsub_new(i), :)*x0_f - y(chsub_new(i), :); 
    if abs(r) < th
        continue; 
    end
    xnum = xnum + 1; 
    [x1, valn, bs] = myFitTchebycheff(X(sub_f,:),y(sub_f, :), x0_f); 

    if valn > th
        sub_f(bs) = []; 
        if numel(sub_f) <= numel(chsub)
            cyc = cyc + 1;
            return; 
        end
        cyc = cyc + 1; 
    else
        x0_f = x1;
        w = valn; 
    end
        
end
end


function [cnt, M] = compute_upper(X, y, x0, valn) % Compute Upper Bound for heuristics
        r = X*x0 - y; 
        M = find(abs(r) > valn)';
        cnt = numel(M); 
end

