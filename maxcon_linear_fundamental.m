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

run('./linearASTAR/sedumi-master/install_sedumi')

addpath(genpath('./data'));
addpath('./utils');
addpath('./linearASTAR');
addpath('./linearASTAR/sedumi');
addpath('./maxcon_BMF');
addpath('./RANSAC');

d = 8;              % model dimention
nsamples = 100;     % number of samples for fourier estimation
nruns = 100;        % number of random runs of the experiment

fundamental_dataset = {'104_108','198_201','417_420','579_582','738_742'};

results = zeros(nruns, length(fundamental_dataset), 14);

for dataset_id = 1:length(fundamental_dataset)
    for i=1:nruns
        [data, th] = read_Fundamental_data(['./data/fundamental/KITTI_',fundamental_dataset{dataset_id},'.mat']);
        [A, b] = genMatrixLinearizeFundamental(data.x1, data.x2);
        Data = [A, b];
        
        nbits = size(Data,1);

        %Shuffle the data
        idx_ = randperm(nbits);
        Data(idx_, :) = Data;

        q = (d+100)/nbits;

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

        % Run Algotim MBF-MaxCon-nL
        bmftic = tic;
        [fixed_bits, results_fevals(i, dataset_id, 1)] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF = toc(bmftic);
        inliers_BMF = setdiff(1:nbits, fixed_bits);
        
        % Run Algotim MBF-MaxCon
        bmfticl = tic;
        [fixed_bits, numTrials] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMFl = toc(bmfticl);
        inliers_BMFl = setdiff(1:nbits, fixed_bits);
        results_fevals(i, dataset_id, 2) = numTrials; 
        
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

        fprintf('(%d, %d)\t Time \t ASTAR: %.3f,\t MaxCon-nL: %.3f,\t MaxCon: %.3f,\t RANSAC: %.3f,\t LORANSAC: %.3f\n', i,dataset_id, t_ASTAR, t_BMF, t_BMFl,t_RANSAC, t_LORANSAC);
        fprintf('(%d, %d)\t Acc \t ASTAR: %d*,\t MaxCon-nL: %d,\t\t MaxCon: %d,\t RANSAC: %d,\t LORANSAC: %d\n', i,dataset_id, numel(inliers_ASTAR), numel(inliers_ASTAR)-numel(inliers_BMF), numel(inliers_ASTAR)-numel(inliers_BMFl), numel(inliers_ASTAR)-numel(inliers_RANSAC), numel(inliers_ASTAR)-numel(inliers_LORANSAC));
        
        results(i, dataset_id,:) = [numel(inliers_ASTAR), t_ASTAR , numel(inliers_BMF), t_BMF, numel(inliers_BMFl), t_BMFl, numel(inliers_RANSAC), t_RANSAC, numel(inliers_LORANSAC), t_LORANSAC, numel(inliers_RANSAC99), t_RANSAC99, numel(inliers_LORANSAC99), t_LORANSAC99];
    end
    
    
end


ASTAR = 1; MaxCon_nL = 3;  MaxCon = 5; RANSAC = 7; LORANSAC= 9; RANSAC99 = 11; LORANSAC99 = 13;

% save('./fundamental_results.mat')
% load('./fundamental_results.mat')

%%%% Information for Table 1 in CVPR paper
fundamental_dataset = {'104_108','198_201','417_420','579_582','738_742'};
fprintf('A*-NAPA-DI, MaxCon-nL, MaxCon, RANSAC, Lo-RANSAC, RANSAC-p, Lo-RANSAC-p \n')
for dataset_id = 1:length(fundamental_dataset)
    res_mean = squeeze(mean(results(:, dataset_id,:), 1));
    fprintf('%10.2f, %10.2f, %10.2f, %10.2f, %10.2f, %10.2f, %10.2f \n', res_mean(ASTAR), res_mean(MaxCon_nL), res_mean(MaxCon) ,res_mean(RANSAC),res_mean(LORANSAC),res_mean(RANSAC99),res_mean(LORANSAC99))
    
    res_max = squeeze(max(results(:, dataset_id,:),[], 1));
    res_min = squeeze(min(results(:, dataset_id,:),[], 1));
    fprintf('(%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f) \n', res_max(ASTAR), res_min(ASTAR), res_max(MaxCon_nL), res_min(MaxCon_nL), res_max(MaxCon), res_min(MaxCon), res_max(RANSAC), res_min(RANSAC), res_max(LORANSAC), res_min(LORANSAC), res_max(RANSAC99), res_min(RANSAC99), res_max(LORANSAC99), res_min(LORANSAC99))
    
    res_meant = squeeze(mean(results(:, dataset_id,:), 1));
    fprintf('%10.2f, %10.2f, %10.2f, %10.2f,  %10.2f, %10.3f, %10.2f \n', res_meant(ASTAR+1), res_meant(MaxCon_nL+1), res_meant(MaxCon+1) ,res_meant(RANSAC+1),res_meant(LORANSAC+1),res_meant(RANSAC99+1),res_meant(LORANSAC99+1))
    % fprintf('\t____________________________________________________________________________\n')
end

%%%% Figure 7 in CVPR paper
MBF_MaxCon_hold = [];
LOR_hold = [];
RAN_hold = [];
for dataset_id = 1:length(fundamental_dataset)
    astar = results(:,dataset_id,1); 
    
    linf = (astar - results(:,dataset_id,MaxCon));
    lor = (astar - results(:,dataset_id,LORANSAC));
    ran = (astar - results(:,dataset_id,RANSAC));
    MBF_MaxCon_hold = [MBF_MaxCon_hold;linf];
    LOR_hold = [LOR_hold;lor];
    RAN_hold = [RAN_hold; ran];

end

histogram(MBF_MaxCon_hold, 'Normalization', 'pdf');hold on
histogram(LOR_hold, 'Normalization', 'pdf');hold on
histogram(RAN_hold, 'Normalization', 'pdf');hold on
legend('MBF-MaxCon','Lo-RANSAC', 'RANSAC', 'location', 'northeast','Interpreter','latex');
xlabel('${\left | \mathcal{I}_{\mathrm{A}^\ast} \right |} - {\left | \mathcal{I}_{\bullet} \right |}  $','Interpreter','latex')
ylabel('Probability Density','Interpreter','latex')
set(gca,'fontsize',18)


%%%% Figure 1 in supplimentary meterials
figure
subplot(1,2,1)
y = results(:,:,1) - results(:,:,[MaxCon_nL,MaxCon]) ;
y = squeeze(mean(y));
bar(1:5, y )
ylabel('$\left | \mathcal{I}_{\mathrm{A}^\ast} \right | - \left | \mathcal{I}_{\bullet} \right |$','Interpreter','latex')
xticklabels({'104-108','198-201','417-420','579-582','738-742'})
legend('MBF-MaxCon-nL','MBF-MaxCon');
set(gca,'fontsize',18)
xtickangle(90)

subplot(1,2,2)
y = results(:,:,[MaxCon_nL+1,MaxCon+1]) ;
y = squeeze(mean(y));
bar(1:5, y )
ylabel('Time (s)','Interpreter','latex')
xticklabels({'104-108','198-201','417-420','579-582','738-742'})
legend('MBF-MaxCon-nL','MBF-MaxCon');
set(gca,'fontsize',18)
xtickangle(90)


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