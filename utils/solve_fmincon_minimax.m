% ##########################################################################
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
% ##########################################################################

function [xopt, u, B] = solve_fmincon_minimax(A, b, c, d, optmincon, x0)

    p = size(A, 2); 
    if nargin < 6
        
        optmincon = optimoptions(@fmincon, 'MaxIter', 500, 'Algorithm','sqp'); 
        optmincon = optimoptions(optmincon,'Display',  'off', 'Diagnostics', 'off');%'iter'); 
        optmincon = optimoptions(optmincon,'GradObj','on','GradConstr','on');%, 'Hessian', 'user-supplied', 'HessFcn', @hessinterior);%, 'TolX', 0, 'TolFun', 0);
        
        nbrimages = size(A, 2); 
        b = A(7:8, :); 
        c = A(9:11, :);
        d = A(end, :); 
        A = reshape(A(1:6, :), [3, 2*nbrimages])'; 
        x0 = [randn(p, 1); 0]; 
    end
  nbrimages = numel(d);  
   
    if~(x0(end))
%         resn = compute_residuals(x0);
        x0(end) = 50; 
    end
    
  options.lb = [-100*ones(1, p), 0];  % Lower bound on the variables.
  options.ub = [+100*ones(1, p), 100];  % Upper bound on the variables.

  [xopt] = fmincon(@objective_fmincon, x0, [], [], [], [], options.lb, options.ub, @constraints_fmincon, optmincon);
  
  res = compute_residuals(xopt); 
%   if min(res) < 0
%       pause; 
%   end
  B = find(abs(res - max(res)) < 0.0001 | res < 0); 
  xopt = xopt(1:p); 
  u = max(res(B)); 
% ----------------------------------------------------------------------
    function [f, g] = objective_fmincon (x)
      f = x(end); 
      g = [zeros(p, 1); 1]; 
       
    end

% % ----------------------------------------------------------------------
% 
%     function h = hessinterior(x, lambda)
%         h = zeros(4); % Hessian of f
% 
%         for i=1:nbrimages
%             r = sqrt((A{i}*x(1:3)+b(:, i))'*(A{i}*x(1:3)+b(:, i)))+eps; 
%             ff = (A{i}*x(1:3) + b(:, i))'*A{i}; 
%             hessc = A{i}'*A{i}/r - 0.5*(ff'*ff)/r^3; % Hessian of i^th constraints
%             h(1:3, 1:3) = h(1:3, 1:3) + lambda.ineqnonlin(i)*hessc;
%             h(4, 1:3) = h(4, 1:3) - lambda.ineqnonlin(i)*c(:, i)';
%         end
%         h = tril(ones(4)).*h;
%     end

% ----------------------------------------------------------------------
    function [cnstr, cnstrequ, J, Jequ] = constraints_fmincon (x)
        
        AxPb = reshape(A*x(1:end-1),2,nbrimages)+b; % Ax + b
        Sqrt_AxPb = sqrt(sum(AxPb.^2));                     % sqrt(Ax + b)
        CxPd = (x(1:end-1)'*c + d);                             % Cx + d          
        cnstr =  Sqrt_AxPb - x(end)*CxPd;
        cnstrequ = []; 
%         SAxPb = sparse(kron(eye(nbrimages),[1 1])).*repmat(AxPb', [1, nbrimages]); 
%         J = reshape(sum(SAxPb*sparse(blkdiag(A{:}))), [3, nbrimages])./repmat(Sqrt_AxPb, [3, 1])-x(4)*c;
%         J2 = [J;-CxPd]; 
        reA = reshape(A', 2*p, nbrimages);
        prm = [1:p; p+1:2*p];
        StA = repmat(AxPb, [p, 1]).* reA(prm(:), :); 
        J = []; 
        for k = 1:p
            J = [J; sum(StA(2*k-1:2*k, :))./(Sqrt_AxPb+eps)]; 
        end
        J = J - x(end)*c;
        J = [J;-CxPd]; 
        Jequ = []; 
    end  

    function resn = compute_residuals(x)
        AxPb = reshape(A*x(1:end-1),2,nbrimages)+b; % Ax + b
        Sqrt_AxPb = sqrt(sum(AxPb.^2));                     % sqrt(Ax + b)
        CxPd = (x(1:end-1)'*c + d)+eps; 
        resn =  zeros(1, nbrimages); 
        id = abs(CxPd) > 0.0001; 
        resn(id) = Sqrt_AxPb(id)./CxPd(id);
        resn(~id) = max(resn); 
    end

end
  



