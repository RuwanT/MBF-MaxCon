clear all;
close all;
warning('off','all')

addpath(genpath('./data'));
addpath('./utils');
addpath('./linearASTAR');
addpath('./linearASTAR/sedumi');
addpath('./maxcon_BMF');
addpath('./RANSAC');

d = 8;
nsamples = 100;           % number of samples for fourier estimation


nruns = 100;

fundamental_dataset = {'104_108','198_201','417_420','579_582','738_742'};

results = zeros(nruns, length(fundamental_dataset), 18);
results_fevals = zeros(nruns, length(fundamental_dataset), 4);

for dataset_id = 1:length(fundamental_dataset)
    for i=1:nruns
        [data, th] = read_Zhipeng_data(['./data/fundamental/KITTI_',fundamental_dataset{dataset_id},'.mat']);
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

        % Run Algotim maxcon-L-inf
        bmftic = tic;
        [fixed_bits, results_fevals(i, dataset_id, 1)] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF = toc(bmftic);
        inliers_BMF = setdiff(1:nbits, fixed_bits);
        
        % Run Algotim maxcon-L-inf with local update
        bmfticl = tic;
        [fixed_bits, numTrials] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, true);
        t_BMFl = toc(bmfticl);
        inliers_BMFl = setdiff(1:nbits, fixed_bits);
        results_fevals(i, dataset_id, 2) = numTrials; 

        % Run algorithm maxcon-max
        bmfmtic = tic;
        fixed_bits = []; % [fixed_bits, results_fevals(i, dataset_id, 3)] = maxcon_BMF('maxcon-max', fittingfn, nbits, minmaxfn, nsamples, q, false);
        t_BMF_max = toc(bmfmtic);
        inliers_BMF_max  = setdiff(1:nbits, fixed_bits);
        
        % Run algorithm maxcon-max with local update
        bmfmticl = tic;
        fixed_bits = []; %[fixed_bits, results_fevals(i, dataset_id, 4)] = maxcon_BMF('maxcon-max', fittingfn, nbits, minmaxfn, nsamples, q, true);
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

        fprintf('(%d, %d)\t Time \t ASTAR: %.3f,\t BMF: %.3f,\t BMFm: %.3f,\t BMFl: %.3f,\t BMFml: %.3f,\t RANSAC: %.3f,\t LORANSAC: %.3f\n', i,dataset_id, t_ASTAR, t_BMF, t_BMF_max, t_BMFl, t_BMF_maxl,t_RANSAC, t_LORANSAC);
        fprintf('(%d, %d)\t Acc \t ASTAR: %d,\t BMF: %d,\t BMFm: %d,\t BMFl: %d,\t BMFml: %d,\t RANSAC: %d,\t LORANSAC: %d\n', i,dataset_id, numel(inliers_ASTAR), numel(inliers_ASTAR)-numel(inliers_BMF), numel(inliers_ASTAR)-numel(inliers_BMF_max), numel(inliers_ASTAR)-numel(inliers_BMFl), numel(inliers_ASTAR)-numel(inliers_BMF_maxl), numel(inliers_ASTAR)-numel(inliers_RANSAC), numel(inliers_ASTAR)-numel(inliers_LORANSAC));
        
        results(i, dataset_id,:) = [numel(inliers_ASTAR), t_ASTAR , numel(inliers_BMF), t_BMF, numel(inliers_BMF_max),  t_BMF_max, numel(inliers_BMFl), t_BMFl, numel(inliers_BMF_maxl), t_BMF_maxl, numel(inliers_RANSAC), t_RANSAC, numel(inliers_LORANSAC), t_LORANSAC, numel(inliers_RANSAC99), t_RANSAC99, numel(inliers_LORANSAC99), t_LORANSAC99];
    end
    
    
end
save('./fundamental_results.mat')

% load('./fundamental_results.mat')

% fprintf('\n')
% fprintf('\t____________________________________________________________________________\n')
% fprintf('\tA*-NAPA-DI \t BMF-max-L \t BMF-maxcon-m \t\t RANSAC \t Lo-RANSAC \t RANSAC-p \t Lo-RANSAC-p \n')
% fprintf('\t____________________________________________________________________________\n')
% for dataset_id = 1:length(fundamental_dataset)
%     res_mean = squeeze(mean(results(:, dataset_id,:), 1));
%     fprintf('\t%10.2f \t %10.2f \t %10.2f \t	%10.2f \t %10.2f %10.2f \t %10.2f \n', res_mean(1), res_mean(7), res_mean(9) ,res_mean(11),res_mean(13),res_mean(15),res_mean(17))
%     
%     res_max = squeeze(max(results(:, dataset_id,:),[], 1));
%     res_min = squeeze(min(results(:, dataset_id,:),[], 1));
%     fprintf('\t(%5.0f-%5.0f) \t (%5.0f-%5.0f) \t (%5.0f-%5.0f) \t	(%5.0f-%5.0f) \t (%5.0f-%5.0f) (%5.0f-%5.0f) \t (%5.0f-%5.0f) \n', res_max(1), res_min(1), res_max(7), res_min(7), res_max(9), res_min(9), res_max(11), res_min(11), res_max(13), res_min(13), res_max(15), res_min(15), res_max(17), res_min(17))
%     
%     res_meant = squeeze(mean(results(:, dataset_id,:), 1));
%     fprintf('\t%10.2f \t %10.2f \t %10.2f \t	%10.2f \t %10.2f %10.3f \t %10.2f \n', res_meant(2), res_meant(8), res_meant(10) ,res_meant(12),res_meant(14),res_meant(16),res_meant(18))
%     fprintf('\t____________________________________________________________________________\n')
% end

fprintf('\n')

fundamental_dataset = {'104_108',};%'198_201','417_420','579_582','738_742'};

fprintf('A*-NAPA-DI, BMF-max-L, BMF-maxcon-m, RANSAC, Lo-RANSAC, RANSAC-p, Lo-RANSAC-p \n')

for dataset_id = 1:length(fundamental_dataset)
    res_mean = squeeze(mean(results(:, dataset_id,:), 1));
    fprintf('%10.2f, %10.2f, %10.2f, %10.2f, %10.2f, %10.2f, %10.2f \n', res_mean(1), res_mean(7), res_mean(9) ,res_mean(11),res_mean(13),res_mean(15),res_mean(17))
    
    res_max = squeeze(max(results(:, dataset_id,:),[], 1));
    res_min = squeeze(min(results(:, dataset_id,:),[], 1));
    fprintf('(%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f), (%5.0f-%5.0f) \n', res_max(1), res_min(1), res_max(7), res_min(7), res_max(9), res_min(9), res_max(11), res_min(11), res_max(13), res_min(13), res_max(15), res_min(15), res_max(17), res_min(17))
    
    res_meant = squeeze(mean(results(:, dataset_id,:), 1));
    fprintf('%10.2f, %10.2f, %10.2f, %10.2f,  %10.2f, %10.3f, %10.2f \n', res_meant(2), res_meant(8), res_meant(10) ,res_meant(12),res_meant(14),res_meant(16),res_meant(18))
    % fprintf('\t____________________________________________________________________________\n')
end



fundamental_dataset = {'104_108','198_201','417_420','579_582','738_742'};
for dataset_id = 1:length(fundamental_dataset)
    % subplot(2,3,dataset_id)
    figure
    linf = results(:,dataset_id,7);
    lor = results(:,dataset_id,13);
    histogram(linf);hold on
    histogram(lor);hold on

    legend('MBF-MaxCon','Lo-RANSAC', 'location', 'northwest','Interpreter','latex');
    xlabel('Number of Inliers','Interpreter','latex')
    ylabel('Frequency','Interpreter','latex')
    % title(['Frame ', fundamental_dataset{dataset_id}], 'Interpreter','latex')
    % ylim([0,2])
    %xlim([0, 45])
    set(gca,'fontsize',18)

end


MBF_MaxCon_hold = [];
LOR_hold = [];
RAN_hold = [];
for dataset_id = 1:length(fundamental_dataset)
    % subplot(2,3,dataset_id)
    %figure
    astar = results(:,dataset_id,1); 
    mean(astar)
    
    linf = (mean(astar) - results(:,dataset_id,7))/mean(astar)*100;
    lor = (mean(astar) - results(:,dataset_id,13))/mean(astar)*100;
    ran = (mean(astar) - results(:,dataset_id,11))/mean(astar)*100;
    MBF_MaxCon_hold = [MBF_MaxCon_hold;linf];
    LOR_hold = [LOR_hold;lor];
    RAN_hold = [RAN_hold; ran];

end


    
histogram(MBF_MaxCon_hold,0:.2:5, 'Normalization', 'pdf');hold on
histogram(LOR_hold,0:.2:5, 'Normalization', 'pdf');hold on
histogram(RAN_hold,0:.2:5, 'Normalization', 'pdf');hold on


legend('MBF-MaxCon','Lo-RANSAC', 'RANSAC', 'location', 'northeast','Interpreter','latex');
xlabel('$\left ({1 - \left | \mathcal{I}_{\bullet} \right |}/{\left | \mathcal{I}_{\mathrm{A}^\ast} \right |}  \right )\times 100$','Interpreter','latex')
ylabel('Probability Density','Interpreter','latex')
% title(['Frame ', fundamental_dataset{dataset_id}], 'Interpreter','latex')
% ylim([0,2])
%xlim([0, 45])
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