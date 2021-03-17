%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maxcon with BMF 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
warning('off','all')

addpath(genpath('../data'));
addpath('../utils');
addpath('../maxcon_BMF');
addpath('../utils');
addpath('../linearASTAR');
addpath('../linearASTAR/sedumi');

% Setup input data
N = [15];     % Number of Points
% sig = 0.03; % Inlier Varience
% osig = 1;   % Outlier Varience
inlNoise_ = 0.1;    % Inlier theshold 
out_var = 5.0;      % Outlier varience

% th = 0.09;  % Inlier Threshold
d = 2;      % Dimension of space
out_frac = .25;  % Outlier fraction

% _____________________________________________________
% {
nruns = 100;
nsamples = [100, 200, 500, 1000, 2500, 5000];
N = 15;
qq = [ 0.2, 0.3, .4, .5, .6, .75];


results = zeros(100, length(nsamples), length(qq));
results_orc = zeros(100, length(nsamples), length(qq));
fevals_orc = zeros(100, length(nsamples), length(qq));

for k = 1:length(qq)
    q = qq(k);
    for j=1:length(nsamples)
        params_fh = struct('method','full-hypercube', 'degree', 0:1);
        params_us = struct('method','uniform-sampling', 'degree', 0:1, 'nsamples', nsamples(j));

        for i=1:nruns
            % Data = generate_linear_data(N, sig, osig, d, out_frac);
            % Data = generate_linear_data_pseudo_outl(N, sig, osig, d, out_frac);
            [x, y, mm, cc, ~, ~, ~,~, th] = genRandomLinearData(N, d, inlNoise_, inlNoise_, out_var, out_frac, 1);
            Data = [x,y];
            nbits = size(Data,1);

            % Query access to boolean function (function defined at the bottom of this file)
            % fittingfn: {F_2}^n -> {-1, 1}
            fittingfn = @(x) bool_function_line_concensus(x, Data, th, nbits, d); % 1: feasible, -1: infeasible

            fourier_coeff_fh = estimate_fourier_coefficients(fittingfn, nbits, 'full-hypercube', params_fh);
            % fourier_coeff_us = estimate_fourier_coefficients(fittingfn, nbits, 'uniform-sampling', params_us);

            influence_fh = fourier_coeff_fh.fc(fourier_coeff_fh.degree==1);
            % influence_us = fourier_coeff_us.fc(fourier_coeff_us.degree==1);
            
            inf_ext = tic;
            [influence_orc, fevlas] = estimate_restricted_degree1_coeff(fittingfn, nbits, [],[], nsamples(j), q, 1:nbits);
            t_infE = toc(inf_ext);
            
            % Run ASTAR to generate ground truth
            tic;
            [sol, outliers_ASTAR, ~,UniqueNodeNumber,hInit, ubInit, levelMax, ~, NODIBP] = ASTARWithH2_DIBP_V7(Data(:,1:d), Data(:, end), rand(d, 1), th, 1,0, 3500);
            x = ones(1,nbits);
            x(outliers_ASTAR) = 0;
            if feval(fittingfn, x) ==-1
                fprintf("trouble")
                continue
            end
            t_ASTAR = toc;
            inliers_ASTAR = setdiff(1:nbits, outliers_ASTAR);
            
            % results(i,j,k) = mean((influence_fh - influence_us).^2);
            results_orc(i,j,k) = mean((influence_fh - influence_orc').^2);
            
            if ~isempty(outliers_ASTAR) &&  ~isempty(inliers_ASTAR)
                min_outiers = min(influence_orc(outliers_ASTAR));
                max_inliers = max(influence_orc(inliers_ASTAR));

                results(i,j,k) = min_outiers - max_inliers;
            else
                results(i,j,k) = 1;
            end
            
            fevals_orc(i,j,k) = fevlas;

            % fprintf('(%d, %d, %d) MSE: %0.5f, %0.5f\n', i,j,k, results(i,j,k), results_orc(i,j,k));
            fprintf('(%d, %d, %d) MSE: %0.5f\n', i,j,k, results_orc(i,j,k));
        end
    end
end

save('ablation_m.mat')
% errorbar(nsamples, mean(results_orc(:,:,1),1), mean(results_orc(:,:,1),1)-quantile(results_orc(:,:,1),.05,1), quantile(results_orc(:,:,1),.95,1)-mean(results_orc(:,:,1),1), 'b-', 'linewidth', 2); hold on
% errorbar(nsamples, mean(results_orc(:,:,2),1), mean(results_orc(:,:,2),1)-quantile(results_orc(:,:,2),.05,1), quantile(results_orc(:,:,2),.95,1)-mean(results_orc(:,:,2),1), 'r-', 'linewidth', 2); hold on
% errorbar(nsamples, mean(results_orc(:,:,3),1), mean(results_orc(:,:,3),1)-quantile(results_orc(:,:,3),.05,1), quantile(results_orc(:,:,3),.95,1)-mean(results_orc(:,:,3),1), 'c-', 'linewidth', 2); hold on
% legend('N=12','N=13','N=15')
% xlabel('Number of samples per data point ($m$)','Interpreter','latex')
% ylabel('Influence Estimation Error','Interpreter','latex')
% set(gca,'fontsize',18) 


cq = parula(6);
for q=3:length(qq)-1
    %errorbar(nsamples, mean(results_orc(:,:,q),1), mean(results_orc(:,:,q),1)-quantile(results_orc(:,:,q),.05,1), quantile(results_orc(:,:,q),.95,1)-mean(results_orc(:,:,q),1),'Color', cq(q,:), 'linewidth', 2); hold on
    plot(nsamples, mean(results_orc(:,:,q),1),'Color', cq(q,:), 'linewidth', 2); hold on
    
end


for q = 1:length(qq)
    figure
    separation = squeeze(results(:,:,q));
    nsamples_val = repmat(nsamples, 100,1);
    separation = separation(:);
    nsamples_val = nsamples_val(:);
    
    h = boxplot(separation,nsamples_val); hold on
    set(h,{'linew'},{2})
    xlabel('Number of samples per data point ($m$)','Interpreter','latex')
    ylabel('Separation','Interpreter','latex')
    set(gca,'fontsize',18)
end


error_hold = [];
nsample_hold = [];
q_hold = [];
for q = 3:length(qq)-1
    
    error_ = squeeze(results_orc(:,:,q));
    nsamples_val = repmat(nsamples, 100,1);
    error_ = error_(:);
    nsamples_val = nsamples_val(:);
    q_val = qq(q)*ones(size(nsamples_val));
    
    error_hold = [error_hold; error_];
    nsample_hold = [nsample_hold; nsamples_val];
    q_hold = [q_hold; q_val];
    
    
end

ps = [1.5,2,2.5,5.5,6,6.5,9.5,10,10.5,13.5,14,14.5,17.5,18,18.5,21.5,22,22.5];
xylabel = q_hold; % need a legend instead, but doesn't appear possible
boxplot(error_hold, {nsample_hold,q_hold} ,'PlotStyle', 'compact','factorgap',0, 'color','krb', 'Positions', ps)
ylim([0,0.006]);
xlim([0,24])
set(gca,'xtick',[2,6,10,14,18,22])
set(gca,'xticklabel',{'100', '200', '500', '1000', '2500', '5000'})
xlabel('Number of samples per data point ($m$)','Interpreter','latex')
ylabel('Influence Estimation Error','Interpreter','latex')
set(gca,'fontsize',18)

%}

% _____________________________________________________
%{
figure
nruns = 100;
nsamples = [5000,];
N = 40;

qq = [ 0.1, 0.2, 0.3, .4, .5, .6, .75, .9];

results = zeros(100, length(nsamples), length(qq));
results_orc = zeros(100, length(nsamples), length(qq));
fevals_orc = zeros(100, length(nsamples), length(qq));
time_orc = zeros(100, length(nsamples), length(qq));

for k = 1:length(qq)
    q = qq(k);
    for j=1:length(nsamples)
        params_fh = struct('method','full-hypercube', 'degree', 0:1);
        params_us = struct('method','uniform-sampling', 'degree', 0:1, 'nsamples', nsamples(j));

        for i=1:nruns
            % Data = generate_linear_data(N, sig, osig, d, out_frac);
            [x, y, mm, cc, ~, ~, ~,~, th] = genRandomLinearData(N, d, inlNoise_, inlNoise_, out_var, out_frac, 1);
            Data = [x,y];
            %Data = generate_linear_data_pseudo_outl(N, sig, osig, d, out_frac);
            nbits = size(Data,1);
            
%             xxx = 0:0.01:1;
%             yyy = mm*xxx + cc;
%             plot(xxx,yyy,'r-');hold on
%             pause(0.1)
            

            % Query access to boolean function (function defined at the bottom of this file)
            % fittingfn: {F_2}^n -> {-1, 1}
            fittingfn = @(x) bool_function_line_concensus(x, Data, th, nbits, d); % 1: feasible, -1: infeasible

            %fourier_coeff_fh = estimate_fourier_coefficients(fittingfn, nbits, 'full-hypercube', params_fh);
            % fourier_coeff_us = estimate_fourier_coefficients(fittingfn, nbits, 'uniform-sampling', params_us);

            %influence_fh = fourier_coeff_fh.fc(fourier_coeff_fh.degree==1);
            % influence_us = fourier_coeff_us.fc(fourier_coeff_us.degree==1);
            
            inf_ext = tic;
            [influence_orc, fevlas] = estimate_restricted_degree1_coeff(fittingfn, nbits, [],[], nsamples(j),q, 1:nbits);
            t_infE = toc(inf_ext);
            
            % Run ASTAR to generate ground truth
            tic;
            [sol, outliers_ASTAR, ~,UniqueNodeNumber,hInit, ubInit, levelMax, ~, NODIBP] = ASTARWithH2_DIBP_V7(Data(:,1:d), Data(:, end), rand(d, 1), th, 1,0, 3500);
            t_ASTAR = toc;
            inliers_ASTAR = setdiff(1:nbits, outliers_ASTAR);
            
            %results_orc(i,j,k) = mean((influence_fh - influence_orc').^2);
            
            if ~isempty(outliers_ASTAR) &&  ~isempty(inliers_ASTAR)
                min_outiers = min(influence_orc(outliers_ASTAR));
                max_inliers = max(influence_orc(inliers_ASTAR));

                results(i,j,k) = min_outiers - max_inliers;
                
            else
                results(i,j,k) = 1;
            end
            
            fevals_orc(i,j,k) = fevlas;
            time_orc(i,j,k) = t_infE;

            fprintf('(%d, %d, %d) MSE: %0.5f, Time: %0.5f (s) \n', i,j,k, results_orc(i,j,k), time_orc(i,j,k));
        end
    end
end

% yyaxis right
% % errorbar(pp,  squeeze(mean(time_orc(:,1,:),1)), squeeze(mean(time_orc(:,1,:),1))-squeeze(quantile(time_orc(:,1,:),.05,1)), squeeze(quantile(time_orc(:,1,:),.95,1))-squeeze(mean(time_orc(:,1,:),1)), 'r-','linewidth', 2, 'HandleVisibility','off'); hold on
% errorbar(pp,  squeeze(mean(time_orc(:,2,:),1)), squeeze(mean(time_orc(:,2,:),1))-squeeze(quantile(time_orc(:,2,:),.05,1)), squeeze(quantile(time_orc(:,2,:),.95,1))-squeeze(mean(time_orc(:,2,:),1)), 'r-','linewidth', 2)
% 
% ylabel('Computation Time (s)','Interpreter','latex')
% 
% yyaxis left
% % errorbar(pp,  squeeze(mean(results_orc(:,1,:),1)), squeeze(mean(results_orc(:,1,:),1))-squeeze(quantile(results_orc(:,1,:),.05,1)), squeeze(quantile(results_orc(:,1,:),.95,1))-squeeze(mean(results_orc(:,1,:),1)), 'b-','linewidth', 2); hold on
% errorbar(pp,  squeeze(mean(results_orc(:,2,:),1)), squeeze(mean(results_orc(:,2,:),1))-squeeze(quantile(results_orc(:,2,:),.05,1)), squeeze(quantile(results_orc(:,2,:),.95,1))-squeeze(mean(results_orc(:,2,:),1)), 'b-','linewidth', 2)
% 
% % legend('$m=500$','$m=1000$','Interpreter','latex')
% xlabel('Sampling Probability ($q$)','Interpreter','latex')
% ylabel('Influence Estimation Error','Interpreter','latex')
% 
% legend('Estimation Error','Computation Time','Interpreter','latex')
% set(gca,'fontsize',18)  

yyaxis left
separation = squeeze(results(:,1,:));
q_val = repmat(qq, 100,1);
separation = separation(:);
q_val = q_val(:);
h = boxplot(separation,q_val); hold on
set(h,{'linew'},{2})
xlabel('Sampling Probability ($q$)','Interpreter','latex')
ylabel('Separation','Interpreter','latex')
set(gca,'fontsize',18)  

yyaxis right
errorbar(1:8,  squeeze(mean(results_orc(:,1,:),1)), squeeze(mean(results_orc(:,1,:),1))-squeeze(quantile(results_orc(:,1,:),.05,1)), squeeze(quantile(results_orc(:,1,:),.95,1))-squeeze(mean(results_orc(:,1,:),1)), 'r--','linewidth', 2)
ylabel('Influence Estimation Error','Interpreter','latex')

save('ablation_q.mat')

%}

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
