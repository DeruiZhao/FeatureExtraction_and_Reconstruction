function U = trussDisp(crossSect)
% geometry
nodeCoordinates = [
    -58.4727    2.55103
    -33.4632    17.5214
    -15.1378    2.86231
    0           18.4901
    15.1378     2.86231
    33.4632     17.5214
    58.4727     2.55103
    ];
elementNodes =[
    1   2
    1   3
    2   3
    2   4
    3   4
    3   5
    4   5
    4   6
    5   6
    5   7
    6   7
    ];
nCoord = size(nodeCoordinates, 1);
nElement = size(elementNodes, 1);
nDof = 2;

% boundary condition
% constraint
supports = zeros(nCoord, nDof);
supports(1, 1 : 2) = 1;
supports(7, 1) = 1;

% load
forces = zeros(nCoord, nDof);
% forces(nCoord, 3) = -1;
forces = [
    0   0
    0	0
    0	0
    0	-2
    0	0
    0	0
    0	0
    ];

% material
E = 1e2;

A = zeros(nElement, 1);
A(1) = crossSect(1);
A(11) = crossSect(1);
A(2) = crossSect(2);
A(10) = crossSect(2);
A(3) = crossSect(3);
A(9) = crossSect(3);
A(4) = crossSect(4);
A(8) = crossSect(4);
A(5) = crossSect(5);
A(7) = crossSect(5);
A(6) = crossSect(6);

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
penalty = 1e15;
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

end
