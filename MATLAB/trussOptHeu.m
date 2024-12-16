clc
clear
close all

%% genetic algorithm

% rng default % for reproducibility

% Define objective function
objective = @(crossSect) objectiveFuncHeu(crossSect);

nCrossSect = 8;
% Define lower and upper bounds for A
lb = ones(nCrossSect, 1) * 0.1; % Lower bound
ub = ones(nCrossSect, 1) * 10; % Upper bound

init = (ones(nCrossSect, 1) * 5)';

options = optimoptions('ga', 'InitialPopulationMatrix', init, 'MaxGenerations', 500, 'OutputFcn', @trussOptHeuFigure);

% Perform optimization using genetic algorithm
[crossSect_opt, fga, flga, oga] = ga(objective, nCrossSect, [], [], [], [], lb, ub, [], options);

disp('Optimized cross sections: ');
disp(crossSect_opt');