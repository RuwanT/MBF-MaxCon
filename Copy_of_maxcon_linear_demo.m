clear all;
close all;
warning('off','all')


addpath(genpath('./data'));
addpath('./maxcon_BMF');
addpath('./utils');
addpath('./linearASTAR');
addpath('./linearASTAR/sedumi');
addpath('./Ransac');

% Setup input data
N = 200;            % Number of Points
d = 8;              % Dimension of space
inlNoise_ = 0.1;    % Inlier theshold 
out_var = 5.0;      % Outlier varience
out_frac = .1;      % Outlier Fraction


nsamples = 100;           % number of samples for fourier estimation'

nruns = 10;
out_fracs =  [ 5, 10, 15, 20, 25, 30, 35, 40 ]/N;

results = zeros(nruns, length(out_fracs), 8);
results_fevals = zeros(nruns, length(out_fracs), 4);

for j=1:length(out_fracs)
    for i=1:nruns
        % Generate Synethetic Data
        [x, y, ~, ~, ~, ~, ~,~, th] = genRandomLinearData(N, d, inlNoise_, inlNoise_, out_var, out_fracs(j), 1);
        Data = [x,y];
        
        nbits = size(Data,1);
        
        %Shuffle the data
        idx_ = randperm(nbits);
        Data(idx_, :) = Data;
        
        q = (d+3)/nbits;
        

        % Query access to boolean function (function defined at the bottom of this file)
        % fittingfn: {F_2}^n -> {-1, 1}
        fittingfn = @(x) bool_function_line_concensus(x, Data, th, nbits, d); % 1: feasible, -1: infeasible

        % Solve minmax problem and return the edge points
        minmaxfn = @(x) solve_minmax(x, Data);


%         % Run ASTAR to generate ground truth
%         tic;
%         [sol, v5, ~,UniqueNodeNumber,hInit, ubInit, levelMax, ~, NODIBP] = ASTARWithH2_DIBP_V7(Data(:,1:d), Data(:, end), rand(d, 1), th, 1,0, 3500);
%         x = ones(1,nbits);
%         x(v5) = 0;
%         if feval(fittingfn, x) ==-1
%             fprintf("trouble")
%             continue
%         end
%         t_ASTAR = toc;
%         inliers_ASTAR = setdiff(1:nbits, v5);

        % Run Algotim maxcon-L-inf without local update
        bmftic = tic;
        [fixed_bits, results_fevals(i, j, 1)] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF = toc(bmftic);
        inliers_BMF = setdiff(1:nbits, fixed_bits);
        
        % Run Algotim maxcon-L-inf with local update
        bmfticl = tic;
        [fixed_bits, numTrials] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMFl = toc(bmfticl);
        inliers_BMFl = setdiff(1:nbits, fixed_bits);
        results_fevals(i, j, 2) = numTrials;

        % Run algorithm maxcon-max without local update
        bmfmtic = tic;
        [fixed_bits, results_fevals(i, j, 3)] = maxcon_BMF('maxcon-max', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF_max = toc(bmfmtic);
        inliers_BMF_max  = setdiff(1:nbits, fixed_bits);
        
        % Run algorithm maxcon-max with local update
        bmfmticl = tic;
        [fixed_bits, results_fevals(i, j, 4)] = maxcon_BMF('maxcon-max', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMF_maxl = toc(bmfmticl);
        inliers_BMF_maxl  = setdiff(1:nbits, fixed_bits);    

        fprintf('(%d, %d)\t Time \t BMF: %.3f,\t BMFm: %.3f,\t BMFl: %.3f,\t BMFml: %.3f\n', i,j, t_BMF, t_BMF_max, t_BMFl, t_BMF_maxl);
        fprintf('(%d, %d)\t Acc \t BMF: %d,\t BMFm: %d,\t BMFl: %d,\t BMFml: %d\n', i,j, numel(inliers_BMF), numel(inliers_BMF_max), numel(inliers_BMFl), numel(inliers_BMF_maxl));
        
        results(i, j,:) = [numel(inliers_BMF), t_BMF, numel(inliers_BMF_max),  t_BMF_max, numel(inliers_BMFl), t_BMFl, numel(inliers_BMF_maxl), t_BMF_maxl ];
    end
end


% Plotting the results
save('./synthetic_run.mat')
% % load('/Users/ruwantennakoon/projects/BMF_david/results_postEccv/synthetic/synthetic_run_loransac.mat')
% 
% iASTAR = 1; iBMF = 3; iBMFm = 5; iBMFl = 7; iBMFml = 9; iRANSAC = 11; iLORANSAC = 13; iRANSAC99 = 15; iLORANSAC99 = 17; itrivial=19; itriviall=21;
% 
% % Ablation BMF Accuracy
% figure
% MBF_MaxCon_nL = results(:, :,iASTAR)- results(:, :,iBMF);
% MBF_MaxCon = results(:, :,iASTAR)- results(:, :,iBMFl);
% MBF_MaxCon_nB = results(:, :,iASTAR)- results(:, :,iBMFml);
% MBF_MaxCon_nR = results(:, :,iASTAR)- results(:, :,itrivial);
% % errorbar(out_fracs*N+.2, mean(MBF_MaxCon_nR,1), mean(MBF_MaxCon_nR,1)-quantile(MBF_MaxCon_nR,.05,1), quantile(MBF_MaxCon_nR,.95,1)-mean(MBF_MaxCon_nR,1), 'k-', 'linewidth', 2); hold on;
% plot(out_fracs*N+.2, mean(MBF_MaxCon_nR,1), 'k-', 'linewidth', 2); hold on;
% 
% errorbar(out_fracs*N-.2, mean(MBF_MaxCon_nB,1), mean(MBF_MaxCon_nB,1)-quantile(MBF_MaxCon_nB,.05,1), quantile(MBF_MaxCon_nB,.95,1)-mean(MBF_MaxCon_nB,1), 'c-', 'linewidth', 2)
% errorbar(out_fracs*N-.4, mean(MBF_MaxCon_nL,1), mean(MBF_MaxCon_nL,1)-quantile(MBF_MaxCon_nL,.05,1), quantile(MBF_MaxCon_nL,.95,1)-mean(MBF_MaxCon_nL,1), 'b-', 'linewidth', 2)
% errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2)
% legend('MBF-MaxCon-nR','MBF-MaxCon-nB','MBF-MaxCon-nL','MBF-MaxCon', 'location', 'northwest','Interpreter','latex');
% xlabel('$N_o$','Interpreter','latex')
% ylabel('$\left | \mathcal{I}_{\mathrm{A}^\ast} \right | - \left | \mathcal{I}_{\bullet} \right |$','Interpreter','latex')
% xlim([0, 45])
% ylim([0, 60])
% set(gca,'fontsize',18)
% 
% % Ablation BMF Time
% figure
% MBF_MaxCon_nL = results(:, :,iBMF+1);
% MBF_MaxCon = results(:, :,iBMFl+1);
% MBF_MaxCon_nB = results(:, :,iBMFm+1);
% MBF_MaxCon_nR = results(:, :,itrivial+1);
% errorbar(out_fracs*N+.2, mean(MBF_MaxCon_nR,1), mean(MBF_MaxCon_nR,1)-quantile(MBF_MaxCon_nR,.05,1), quantile(MBF_MaxCon_nR,.95,1)-mean(MBF_MaxCon_nR,1), 'k-', 'linewidth', 2); hold on;
% errorbar(out_fracs*N-.2, mean(MBF_MaxCon_nB,1), mean(MBF_MaxCon_nB,1)-quantile(MBF_MaxCon_nB,.05,1), quantile(MBF_MaxCon_nB,.95,1)-mean(MBF_MaxCon_nB,1), 'c-', 'linewidth', 2)
% errorbar(out_fracs*N-.4, mean(MBF_MaxCon_nL,1), mean(MBF_MaxCon_nL,1)-quantile(MBF_MaxCon_nL,.05,1), quantile(MBF_MaxCon_nL,.95,1)-mean(MBF_MaxCon_nL,1), 'b-', 'linewidth', 2)
% errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2)
% legend('MBF-MaxCon-nR','MBF-MaxCon-nB','MBF-MaxCon-nL','MBF-MaxCon', 'location', 'northwest','Interpreter','latex');
% xlabel('$N_o$','Interpreter','latex')
% ylabel('Time (s)','Interpreter','latex')
% xlim([0, 45])
% ylim([0, 50])
% set(gca,'fontsize',18)
% 
% % Comparison Accuracy
% figure
% 
% MBF_MaxCon = results(:, :,iASTAR)- results(:, :,iBMFl);
% Ransac = results(:, :,iASTAR)- results(:, :,iRANSAC);
% Lo_Ransac = results(:, :,iASTAR)- results(:, :,iLORANSAC);
% 
% errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2);hold on
% errorbar(out_fracs*N, mean(Ransac,1), mean(Ransac,1)-quantile(Ransac,.05,1), quantile(Ransac,.95,1)-mean(Ransac,1), 'b-', 'linewidth', 2)
% errorbar(out_fracs*N, mean(Lo_Ransac,1), mean(Lo_Ransac,1)-quantile(Lo_Ransac,.05,1), quantile(Lo_Ransac,.95,1)-mean(Lo_Ransac,1), 'k-', 'linewidth', 2)
% 
% legend('MBF-MaxCon','RANSAC','Lo-RANSAC', 'location', 'northwest','Interpreter','latex');
% xlabel('$N_o$','Interpreter','latex')
% ylabel('$\left | \mathcal{I}_{\mathrm{A}^\ast} \right | - \left | \mathcal{I}_{\bullet} \right |$','Interpreter','latex')
% xlim([0, 45])
% % ylim([0, 60])
% set(gca,'fontsize',18)
% 
% % Comparison Accuracy
% figure
% Astar = results(:, :,iASTAR+1);
% MBF_MaxCon = results(:, :,iBMFl+1);
% Ransac = results(:, :,iRANSAC+1);
% Lo_Ransac = results(:, :,iLORANSAC+1);
% 
% errorbar(out_fracs*N, mean(Astar,1), mean(Astar,1)-quantile(Astar,.05,1), quantile(Astar,.95,1)-mean(Astar,1), 'g-', 'linewidth', 2);hold on
% errorbar(out_fracs*N, mean(Ransac,1), mean(Ransac,1)-quantile(Ransac,.05,1), quantile(Ransac,.95,1)-mean(Ransac,1), 'b-', 'linewidth', 2)
% errorbar(out_fracs*N, mean(Lo_Ransac,1), mean(Lo_Ransac,1)-quantile(Lo_Ransac,.05,1), quantile(Lo_Ransac,.95,1)-mean(Lo_Ransac,1), 'k-', 'linewidth', 2)
% errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2);hold on
% 
% legend('$\textrm{A}^\ast$-NAPA-DIBP','RANSAC','Lo-RANSAC', 'MBF-MaxCon', 'location', 'northwest','Interpreter','latex');
% xlabel('$N_o$','Interpreter','latex')
% ylabel('Time (s)','Interpreter','latex')
% xlim([0, 45])
% ylim([0, 700])
% set(gca,'fontsize',18)



%%%%%%%% Bool functions of interest %%%%%%%%%%%
function f = bool_function_line_concensus(x, data, th, nbits, d)
    % f: {F_2}^n -> {-1, 1}
    % Inputs: 
    % x:    input vector of nbits
    % Output:
    % 1: feasible, -1: infeasible
    
    x = x > 0.5;    % may only work for 0, 1 dictionary 
    if sum(x) < d + 1
        f = int8(1);
    else
        f = int8(myFitTchebycheff_dist(data(x,:)') < th)*2 - int8(1);
    end
end

function sol = solve_minmax(x, data)
    [~, d, basis ] = myFitTchebycheff(data(x, :)');
    sol.basis = basis;
    sol.d = d;

end