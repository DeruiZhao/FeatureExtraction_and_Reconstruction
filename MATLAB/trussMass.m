function mass = trussMass(crossSect)
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

A = zeros(nElement, 1);
A(1) = crossSect(1);
A(2) = crossSect(2);
A(3) = crossSect(3);
A(4) = crossSect(4);
A(5) = crossSect(5);
A(6) = crossSect(6);
A(7) = crossSect(7);
A(8) = crossSect(8);

% length of each element
L = zeros(nElement, 1);
for i = 1 : nElement
    xx = nodeCoordinates(elementNodes(i, 2), 1) - nodeCoordinates(elementNodes(i, 1), 1);
    yy = nodeCoordinates(elementNodes(i, 2), 2) - nodeCoordinates(elementNodes(i, 1), 2);
    L(i) = sqrt(xx^2 + yy^2);
end

mass = A' * L;
end
