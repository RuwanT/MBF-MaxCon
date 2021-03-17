% solve minimax for pseudo-convex problems
function [sol, fval, idxActive] = minimaxPseudo2(A,b,c,d,solInit,th,d1)
options = optimoptions('fminimax','Display','off', 'ConstraintTolerance', 1e-6, 'FunctionTolerance',  1e-6, 'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-6, 'TolConSQP', 1e-6,'TolConSQP', 1e-6); % Minimize abs. values

[sol, res, fval, ~] = fminimax(@resEval, solInit, [], [], [], [], [], [], [], options);
idxActive = find(abs(res-fval)<1e-6);
if numel(idxActive) == 0
    idxActive = find(abs(res-max(res))<1e-6);
end

    function f = resEval(x)
        f = resPseudoConvex(A,b,c,d,x,th,d1);
    end
end