%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maxcon with BMF 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
warning('off','all')

addpath('../utils');

% Setup input data
N = [12];     % Number of Points
sig = 0.03; % Inlier Varience
osig = 1;   % Outlier Varience
th = 0.09;  % Inlier Threshold
d = 2;      % Dimension of space
out_frac = .3;  % Outlier fraction

% Data = generate_linear_data(N, sig, osig, d, out_frac);
Data = generate_linear_data_pseudo_outl(N, sig, osig, d, out_frac);
nbits = size(Data,1);

% Query access to boolean function (function defined at the bottom of this file)
% fittingfn: {F_2}^n -> {-1, 1}
fittingfn = @(x) bool_function_line_concensus(x, Data, th, nbits, d); % 1: feasible, -1: infeasible

% Setup fourier estimation
% Available methods: 'full_hypercube', 'uniform_sampling', 'Goldreich_Levin', 'MBF_ODonnell_2005'
% Method specific parameters

params_fh = struct('method','full-hypercube', 'degree', 0:12);
params_us = struct('method','uniform-sampling', 'degree', 0:12, 'nsamples', 5000);
params_GL = struct('method','Goldreich-Levin', 'nsamples', 500, 'T', 0.05^2);
% C on OD changes with nbits, at the moment I dont have any idea on how to adjust it automatically.
params_OD = struct('method','MBF-ODonnell-2005', 'nsamples', 5000, 'epsilon', 0.1, 'C', 1.1);   
params_ui = struct('method','uniform-influence', 'nsamples', 5000);

% method_params = { params_fh, params_us, params_GL, params_OD};
method_params = {  params_fh, params_us, params_GL, params_OD};

%prepare display env
outl_s = [100, 100, 200, 200];
outl_disp = {'r.','mo','g^','ks'};
fs_disp = {'r-', 'b-', 'g-', 'k-'};

subplot(1,2,1)
scatter(Data(:,1), Data(:,3), 10, 'b','filled');
hold on;
ylim([min(0, min(Data(:,3))),max(1, max(Data(:,3)))])
legend_plt1 = {};
legend_plt1(1) = {'Data'};
legend_plt2 = {};

% Run estimation and result calculation
max_con_1 = [];
for i = 1:length(method_params)
    disp(' ')
    disp(['Running method: ', method_params{i}.method])
    t = cputime;
    fourier_coeff_ = estimate_fourier_coefficients(fittingfn, nbits, method_params{i}.method, method_params{i});
    e = cputime-t;
    disp(['Run time : ', num2str(e), 's'])
    disp(['Percentage coefficients estimated : ', num2str(length(fourier_coeff_.fc)/(2^nbits)*100), '%'])
    
    rhd_ = relative_hamming_distance(fourier_coeff_, fittingfn, nbits, 'uniform_sampling', 5000);
    disp(['Relative Hamming Distance : ', num2str(rhd_) ])
    
    subplot(1,2,1)
    [maxcon_set, outls] = estimate_largest_feaseable_set(fourier_coeff_, nbits); % ** inefficient code ***
    % [maxcon_set, outls] = estimate_maxcon_set_influence(fittingfn, fourier_coeff_, nbits);
    scatter(Data(outls,1), Data(outls,3), outl_s(i), outl_disp{i});
    legend_plt1(i+1) = {method_params{i}.method};
    if strcmp(method_params{i}.method, method_params{1}.method)
        max_con_1 = maxcon_set;
        fourier_coeff = fourier_coeff_;
    end
    
    subplot(1,2,2)
    plot_fourier_spectrum(fourier_coeff_, nbits, fs_disp{i}); hold on
    legend_plt2(i) = {method_params{i}.method};

end

subplot(1,2,1)
legend(legend_plt1,'Location','southeast' )
if ~isempty(max_con_1)
    [theta,d,~] = myFitTchebycheff(Data(max_con_1,:)');
    plot_maxcon_hypo(theta, th, 'r--');
end
set(gca,'fontsize',18)

subplot(1,2,2)
legend(legend_plt2)
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
