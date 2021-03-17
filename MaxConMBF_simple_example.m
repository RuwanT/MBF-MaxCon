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
N = 50;            % Number of Points
d = 2;              % Dimension of space
inlNoise_ = 0.5;    % Inlier theshold 
out_var = 5.0;      % Outlier varience
out_frac = .3;      % Outlier Fraction

nsamples = 100;           % number of samples for fourier estimation'


% Generate Synethetic Data
[x, y, ~, ~, ~, ~, ~,~, th] = genRandomLinearData(N, d, inlNoise_, inlNoise_, out_var, out_frac, 1);
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


% Run MBF-MaxCon
bmfticl = tic;
[outliers_BMFl, numTrials] = maxcon_BMF('maxcon-Linf', fittingfn, nbits, minmaxfn, nsamples, q, true);
t_BMFl = toc(bmfticl);
inliers_BMFl = setdiff(1:nbits, outliers_BMFl);
    
% plot inliers and outlier from MBF-MaxCon
scatter(Data(inliers_BMFl,1), Data(inliers_BMFl,3), 25, 'b','filled'); hold on
scatter(Data(outliers_BMFl,1), Data(outliers_BMFl,3), 25, 'r','filled')

% plot the model for the inlier set
[theta,d,~] = myFitTchebycheff(Data(inliers_BMFl,:)');
x = [-1,1;1,1];
y = x*theta;
pl = plot(x(:,1), y+th, 'r--','HandleVisibility','off');
plot(x(:,1), y-th, 'r--','HandleVisibility','off')

legend('Inliers by MBF-MaxCon','Outliers by MBF-MaxCon')
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