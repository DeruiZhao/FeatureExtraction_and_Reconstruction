clc
clear
close all

%% PRE-PROCESSING
% geometry
nodeCoordinates = [
    8.34e-01    2.47e00
    8.34e-01    4.75e01
    2.76e01     2.50e01
    4.20e01     4.64e01
    4.20e01     3.59e00
    7.77e01     2.50e01
    ];
elementNodes =[
    1   3
    1   5
    2   3
    2   4
    3   4
    3   5
    4   6
    5   6
    ];
nCoord = size(nodeCoordinates, 1);
nElement = size(elementNodes, 1);
nDof = 2;

% boundary condition
% constraint
supports = zeros(nCoord, nDof);
supports(1 : 2, 1 : 2) = 1;

% load
forces = zeros(nCoord, nDof);
forces = [
    0   0
    0	0
    0	0
    0	0
    0	0
    0	-1
    ];

% material
E = 1e2;

% crossSect = 5 * ones(6, 1)
crossSect = [
    % 3.5
    % 7
    % 3.5
    % 7
    % 2.5
    % 3
    % 5
    % 5
    4.5711
    7.4394
    4.3578
    7.3643
    3.8569
    3.9131
    5.6240
    5.8502
    ];

A = zeros(nElement, 1);
A(1) = crossSect(1);
A(2) = crossSect(2);
A(3) = crossSect(3);
A(4) = crossSect(4);
A(5) = crossSect(5);
A(6) = crossSect(6);
A(7) = crossSect(7);
A(8) = crossSect(8);

%% SOLVER

% length of each element
L = zeros(nElement, 1);
for i = 1 : nElement
    xx = nodeCoordinates(elementNodes(i, 2), 1) - nodeCoordinates(elementNodes(i, 1), 1);
    yy = nodeCoordinates(elementNodes(i, 2), 2) - nodeCoordinates(elementNodes(i, 1), 2);
    L(i) = sqrt(xx^2 + yy^2);
end

% cosine
cosine = zeros(nElement, 2);
for i = 1 : nElement
    xx = nodeCoordinates(elementNodes(i, 2), 1) - nodeCoordinates(elementNodes(i, 1), 1);
    yy = nodeCoordinates(elementNodes(i, 2), 2) - nodeCoordinates(elementNodes(i, 1), 2);
    cosine(i, 1) = xx / L(i);
    cosine(i, 2) = yy / L(i);
end

% stiffness matrix
K = zeros(nCoord * nDof);
for i = 1 : nElement
    KE = E * A(i) / L(i) * [
        cosine(i, 1)^2, cosine(i, 1) * cosine(i, 2), -cosine(i, 1)^2, -cosine(i, 1) * cosine(i, 2);
        cosine(i, 1) * cosine(i, 2), cosine(i, 2)^2, -cosine(i, 1) * cosine(i, 2), -cosine(i, 2)^2;
        -cosine(i, 1)^2, -cosine(i, 1) * cosine(i, 2), cosine(i, 1)^2, cosine(i, 1) * cosine(i, 2);
        -cosine(i, 1) * cosine(i, 2), -cosine(i, 2)^2, cosine(i, 1) * cosine(i, 2), cosine(i, 2)^2;
        ];
    elementDof = [
        2 * elementNodes(i, 1) - 1
        2 * elementNodes(i, 1)
        2 * elementNodes(i, 2) - 1
        2 * elementNodes(i, 2)
        ];
    K(elementDof, elementDof) = K(elementDof, elementDof) + KE;
end

% update the stiffness matrix
penalty = 1e10;
for i = 1 : nCoord
    if supports(i, 1) == 1
        K(2 * i - 1, 2 * i - 1) = K(2 * i - 1, 2 * i - 1) + penalty;
    end
    if supports(i, 2) == 1
        K(2 * i, 2 * i) = K(2 * i, 2 * i) + penalty;
    end
end

% displacement
load = reshape(forces', nCoord * nDof, 1);
U = K \ load;

U = reshape(U, nDof, nCoord)';

% stress
stress = zeros(nElement, 1);
for i = 1 : nElement
    xx = U(elementNodes(i, 2), 1) - U(elementNodes(i, 1), 1);
    yy = U(elementNodes(i, 2), 2) - U(elementNodes(i, 1), 2);
    stress(i) = E / L(i) * (cosine(i, 1) * xx + cosine(i, 2) * yy);
end

%% POST-PROCESSING

set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultTextFontSize', 12);

% Plot the structure with original and deformed shapes
figure(1);
hold on;

% Original structure
for i = 1 : size(elementNodes, 1)
    node1 = elementNodes(i, 1);
    node2 = elementNodes(i, 2);
    x = [nodeCoordinates(node1, 1), nodeCoordinates(node2, 1)];
    y = [nodeCoordinates(node1, 2), nodeCoordinates(node2, 2)];
    plot(x, y, '--g', 'LineWidth', 1); % Original shape in dashed green
end

% Deformed structure
scale = 1; % Scale for deformation
nU = nodeCoordinates + scale * U;
for i = 1 : size(elementNodes, 1)
    node1 = elementNodes(i, 1);
    node2 = elementNodes(i, 2);
    x = [nU(node1, 1), nU(node2, 1)];
    y = [nU(node1, 2), nU(node2, 2)];
    plot(x, y, 'b-', 'LineWidth', 2); % Deformed shape in blue
end

% set(gcf, 'Units', 'inches', 'Position', [1, 1, 12, 3]);

% xlim([-65, 65]);
% ylim([-5, 25]);

grid on;
axis equal;
hold off;

% Plot the structure with section areas
figure(2);
hold on;

% Get stress range
minStress = min(stress);
maxStress = max(stress);

% Generate stress color map
% stressColors = colormap(jet); % Default colormap
stressColors = flipud(turbo);
colormap(stressColors);
caxis([minStress, maxStress]); % Set color axis limits

for i = 1 : size(elementNodes, 1)
    node1 = elementNodes(i, 1);
    node2 = elementNodes(i, 2);
    x = [nU(node1, 1), nU(node2, 1)];
    y = [nU(node1, 2), nU(node2, 2)];
    colorIndex = round((stress(i) - minStress) / (maxStress - minStress) * (size(stressColors, 1) - 1)) + 1;
    % Line thickness proportional to area
    plot(x, y, 'Color', stressColors(colorIndex, :), 'LineWidth', 30 * A(i) / max(A));
end

c = colorbar;
c.Label.String = 'Stress (GPa)';
c.Label.FontSize = 12;

% set(gcf, 'Units', 'inches', 'Position', [1, 1, 12, 3]);

% xlim([-65, 65]);
% ylim([-5, 25]);

% grid on;
axis equal;
hold off;