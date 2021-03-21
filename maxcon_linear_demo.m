%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consensus Maximisation Using Influences of Monotone Boolean Functions
% Proposed in
% Tennakoon, Ruwan and Suter, David and Zhang, Erchuan and Chin, Tat-Jun 
% and Bab-Hadiashar, Alireza "Consensus Maximisation Using Influences of 
% Monotone Boolean Functions"
% In Proceedings of the IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2021.
% 
% Copyright (c) 2021 Ruwan Tennakoon (ruwan.tennakoon@rmit.edu.au)
% School of Computing Technologies, RMIT University, Australia
% https://ruwant.github.io/
% Please acknowledge the authors by citing the above paper in any academic 
% publications that have made use of this package or part of it.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
warning('off','all')

addpath(genpath('./data'));
addpath('./maxcon_BMF');
addpath('./utils');
addpath('./linearASTAR');
addpath('./linearASTAR/sedumi');
addpath('./Ransac');

% Setup input synthetic data
N = 200;            % Number of Points
d = 8;              % Dimension of space
inlNoise_ = 0.1;    % Inlier theshold 
out_var = 5.0;      % Outlier varience


out_fracs =  [ 5, 10, 15, 20, 25, 30, 35, 40 ]/N;

nsamples = 100;     % number of samples for fourier estimation
nruns = 100;        % number of random runs f the experiment

%Hold results
results = zeros(nruns, length(out_fracs), 22);
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


        % Run ASTAR to generate ground truth
        tic;
        [sol, v5, ~,UniqueNodeNumber,hInit, ubInit, levelMax, ~, NODIBP] = ASTARWithH2_DIBP_V7(Data(:,1:d), Data(:, end), rand(d, 1), th, 1,0, 3500);
        x = ones(1,nbits);
        x(v5) = 0;
        if feval(fittingfn, x) ==-1
            fprintf("trouble")
            continue
        end
        t_ASTAR = toc;
        inliers_ASTAR = setdiff(1:nbits, v5);

        %  Run Algotim MBF-MaxCon-nL
        bmftic = tic;
        [fixed_bits, results_fevals(i, j, 1)] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF = toc(bmftic);
        inliers_BMF = setdiff(1:nbits, fixed_bits);
        
        %  Run Algotim MBF-MaxCon
        bmfticl = tic;
        [fixed_bits, numTrials] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMFl = toc(bmfticl);
        inliers_BMFl = setdiff(1:nbits, fixed_bits);
        results_fevals(i, j, 2) = numTrials;

        %  Run Algotim MBF-MaxCon-nB-nL
        bmfmtic = tic;
        [fixed_bits, results_fevals(i, j, 3)] = maxcon_BMF('maxcon-max', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF_max = toc(bmfmtic);
        inliers_BMF_max  = setdiff(1:nbits, fixed_bits);
        
        % Run Algotim MBF-MaxCon-nB
        bmfmticl = tic;
        [fixed_bits, results_fevals(i, j, 4)] = maxcon_BMF('maxcon-max', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMF_maxl = toc(bmfmticl);
        inliers_BMF_maxl  = setdiff(1:nbits, fixed_bits);
        
        % Run algorithm RANSAC
        ransactic = tic;
        [~, inliers, ~] = runRANSAC(Data', th, nbits, d, numTrials, 0, 1, t_BMFl);
        t_RANSAC = toc(ransactic);
        inliers_RANSAC  = inliers;
        
        % Run algorithm LORANSAC
        loransactic = tic;
        [~, inliers, ~] = runLoRANSAC(Data', th, nbits, d, numTrials, 0, 1, t_BMFl);
        t_LORANSAC = toc(loransactic);
        inliers_LORANSAC  = inliers;
        
        % Run algorithm RANSAC with .99 prob
        ransactic99 = tic;
        [~, inliers, ~] = runRANSAC(Data', th, nbits, d, numTrials, 1, 1, t_BMFl);
        t_RANSAC99 = toc(ransactic99);
        inliers_RANSAC99  = inliers;
        
        % Run algorithm LORANSAC with .99 prob
        loransactic99 = tic;
        [~, inliers, ~] = runLoRANSAC(Data', th, nbits, d, numTrials, 1, 1, t_BMFl);
        t_LORANSAC99 = toc(loransactic99);
        inliers_LORANSAC99  = inliers;
        
        
        % Run Algotim MBF-MaxCon-nR-nL
        trivialtic = tic;
        [fixed_bits, ~] = maxcon_BMF_trivial('', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF_trivial = toc(trivialtic);
        inliers_BMF_trivial  = setdiff(1:nbits, fixed_bits);
        
        % Run Algotim MBF-MaxCon-nR-nL
        trivialticl = tic;
        [fixed_bits, ~] = maxcon_BMF_trivial('', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMF_triviall = toc(trivialticl);
        inliers_BMF_triviall  = setdiff(1:nbits, fixed_bits);
        

        fprintf('(%d, %d)\t Time \t ASTAR: %.3f,\t MBF-MaxCon-nL: %.3f,\t MBF-MaxCon-nB-nL: %.3f,\t MBF-MaxCon: %.3f,\t MBF-MaxCon-nB: %.3f,\t RANSAC: %.3f,\t LORANSAC: %.3f\n', i,j, t_ASTAR, t_BMF, t_BMF_max, t_BMFl, t_BMF_maxl,t_RANSAC, t_LORANSAC);
        fprintf('(%d, %d)\t Acc \t ASTAR: %d,\t MBF-MaxCon-nL: %d,\t MBF-MaxCon-nB-nL: %d,\t\t MBF-MaxCon: %d,\t\t MBF-MaxCon-nB: %d,\t RANSAC: %d,\t LORANSAC: %d\n', i,j, numel(inliers_ASTAR), numel(inliers_ASTAR)-numel(inliers_BMF), numel(inliers_ASTAR)-numel(inliers_BMF_max), numel(inliers_ASTAR)-numel(inliers_BMFl), numel(inliers_ASTAR)-numel(inliers_BMF_maxl), numel(inliers_ASTAR)-numel(inliers_RANSAC), numel(inliers_ASTAR)-numel(inliers_LORANSAC));
        
        results(i, j,:) = [numel(inliers_ASTAR), t_ASTAR , numel(inliers_BMF), t_BMF, numel(inliers_BMF_max),  t_BMF_max, numel(inliers_BMFl), t_BMFl, numel(inliers_BMF_maxl), t_BMF_maxl, numel(inliers_RANSAC), t_RANSAC, numel(inliers_LORANSAC), t_LORANSAC, numel(inliers_RANSAC99), t_RANSAC99, numel(inliers_LORANSAC99), t_LORANSAC99, numel(inliers_BMF_trivial), t_BMF_trivial, numel(inliers_BMF_triviall), t_BMF_triviall];
    end
end


% Plotting the results
% save('./synthetic_run.mat')
% load('./synthetic_run.mat')

iASTAR = 1; iBMF = 3; iBMFm = 5; iBMFl = 7; iBMFml = 9; iRANSAC = 11; iLORANSAC = 13; iRANSAC99 = 15; iLORANSAC99 = 17; itrivial=19; itriviall=21;

% Figure 5 (a) of CVPR paper
% Results of the ablation study for 8-dimensional robust linear regression with synthetic data (a) Number of
% inliers found compared with the global optimal (obtained using A*).
using
figure
MBF_MaxCon_nL = results(:, :,iASTAR)- results(:, :,iBMF);
MBF_MaxCon = results(:, :,iASTAR)- results(:, :,iBMFl);
MBF_MaxCon_nB = results(:, :,iASTAR)- results(:, :,iBMFml);
MBF_MaxCon_nR = results(:, :,iASTAR)- results(:, :,itrivial);
% errorbar(out_fracs*N+.2, mean(MBF_MaxCon_nR,1), mean(MBF_MaxCon_nR,1)-quantile(MBF_MaxCon_nR,.05,1), quantile(MBF_MaxCon_nR,.95,1)-mean(MBF_MaxCon_nR,1), 'k-', 'linewidth', 2); hold on;
plot(out_fracs*N+.2, mean(MBF_MaxCon_nR,1), 'k-', 'linewidth', 2); hold on;

errorbar(out_fracs*N-.2, mean(MBF_MaxCon_nB,1), mean(MBF_MaxCon_nB,1)-quantile(MBF_MaxCon_nB,.05,1), quantile(MBF_MaxCon_nB,.95,1)-mean(MBF_MaxCon_nB,1), 'c-', 'linewidth', 2)
errorbar(out_fracs*N-.4, mean(MBF_MaxCon_nL,1), mean(MBF_MaxCon_nL,1)-quantile(MBF_MaxCon_nL,.05,1), quantile(MBF_MaxCon_nL,.95,1)-mean(MBF_MaxCon_nL,1), 'b-', 'linewidth', 2)
errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2)
legend('MBF-MaxCon-nR','MBF-MaxCon-nB','MBF-MaxCon-nL','MBF-MaxCon', 'location', 'northwest','Interpreter','latex');
xlabel('$N_o$','Interpreter','latex')
ylabel('$\left | \mathcal{I}_{\mathrm{A}^\ast} \right | - \left | \mathcal{I}_{\bullet} \right |$','Interpreter','latex')
xlim([0, 45])
ylim([0, 60])
set(gca,'fontsize',18)

% Figure 5 (b) of CVPR paper
% Results of the ablation study for 8-dimensional robust linear regression with synthetic data (b) Variation of computational time with
% number of outliers.
figure
MBF_MaxCon_nL = results(:, :,iBMF+1);
MBF_MaxCon = results(:, :,iBMFl+1);
MBF_MaxCon_nB = results(:, :,iBMFm+1);
MBF_MaxCon_nR = results(:, :,itrivial+1);
errorbar(out_fracs*N+.2, mean(MBF_MaxCon_nR,1), mean(MBF_MaxCon_nR,1)-quantile(MBF_MaxCon_nR,.05,1), quantile(MBF_MaxCon_nR,.95,1)-mean(MBF_MaxCon_nR,1), 'k-', 'linewidth', 2); hold on;
errorbar(out_fracs*N-.2, mean(MBF_MaxCon_nB,1), mean(MBF_MaxCon_nB,1)-quantile(MBF_MaxCon_nB,.05,1), quantile(MBF_MaxCon_nB,.95,1)-mean(MBF_MaxCon_nB,1), 'c-', 'linewidth', 2)
errorbar(out_fracs*N-.4, mean(MBF_MaxCon_nL,1), mean(MBF_MaxCon_nL,1)-quantile(MBF_MaxCon_nL,.05,1), quantile(MBF_MaxCon_nL,.95,1)-mean(MBF_MaxCon_nL,1), 'b-', 'linewidth', 2)
errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2)
legend('MBF-MaxCon-nR','MBF-MaxCon-nB','MBF-MaxCon-nL','MBF-MaxCon', 'location', 'northwest','Interpreter','latex');
xlabel('$N_o$','Interpreter','latex')
ylabel('Time (s)','Interpreter','latex')
xlim([0, 45])
ylim([0, 50])
set(gca,'fontsize',18)

% Figure 6 (a) of CVPR paper
% Results for 8 dimensional robust linear regression
% with synthetic data (a) Number of inliers found compared
% with the global optimal (obtained using A)
figure

MBF_MaxCon = results(:, :,iASTAR)- results(:, :,iBMFl);
Ransac = results(:, :,iASTAR)- results(:, :,iRANSAC);
Lo_Ransac = results(:, :,iASTAR)- results(:, :,iLORANSAC);

errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2);hold on
errorbar(out_fracs*N, mean(Ransac,1), mean(Ransac,1)-quantile(Ransac,.05,1), quantile(Ransac,.95,1)-mean(Ransac,1), 'b-', 'linewidth', 2)
errorbar(out_fracs*N, mean(Lo_Ransac,1), mean(Lo_Ransac,1)-quantile(Lo_Ransac,.05,1), quantile(Lo_Ransac,.95,1)-mean(Lo_Ransac,1), 'k-', 'linewidth', 2)

legend('MBF-MaxCon','RANSAC','Lo-RANSAC', 'location', 'northwest','Interpreter','latex');
xlabel('$N_o$','Interpreter','latex')
ylabel('$\left | \mathcal{I}_{\mathrm{A}^\ast} \right | - \left | \mathcal{I}_{\bullet} \right |$','Interpreter','latex')
xlim([0, 45])
% ylim([0, 60])
set(gca,'fontsize',18)

% Figure 6 (a) of CVPR paper
% Results for 8 dimensional robust linear regression
% with synthetic data (b) Variation
% of computational time with number of outliers.
figure
Astar = results(:, :,iASTAR+1);
MBF_MaxCon = results(:, :,iBMFl+1);
Ransac = results(:, :,iRANSAC+1);
Lo_Ransac = results(:, :,iLORANSAC+1);

errorbar(out_fracs*N, mean(Astar,1), mean(Astar,1)-quantile(Astar,.05,1), quantile(Astar,.95,1)-mean(Astar,1), 'g-', 'linewidth', 2);hold on
errorbar(out_fracs*N, mean(Ransac,1), mean(Ransac,1)-quantile(Ransac,.05,1), quantile(Ransac,.95,1)-mean(Ransac,1), 'b-', 'linewidth', 2)
errorbar(out_fracs*N, mean(Lo_Ransac,1), mean(Lo_Ransac,1)-quantile(Lo_Ransac,.05,1), quantile(Lo_Ransac,.95,1)-mean(Lo_Ransac,1), 'k-', 'linewidth', 2)
errorbar(out_fracs*N, mean(MBF_MaxCon,1), mean(MBF_MaxCon,1)-quantile(MBF_MaxCon,.05,1), quantile(MBF_MaxCon,.95,1)-mean(MBF_MaxCon,1), 'r-', 'linewidth', 2);hold on

legend('$\textrm{A}^\ast$-NAPA-DIBP','RANSAC','Lo-RANSAC', 'MBF-MaxCon', 'location', 'northwest','Interpreter','latex');
xlabel('$N_o$','Interpreter','latex')
ylabel('Time (s)','Interpreter','latex')
xlim([0, 45])
ylim([0, 700])
set(gca,'fontsize',18)



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