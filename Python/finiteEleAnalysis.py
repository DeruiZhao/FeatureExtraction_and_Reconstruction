from math import pi
from turtle import color
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.tri import Triangulation
from regex import R
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from shapely import LineString
import pandas as pd
import itertools
import networkx as nx
from itertools import combinations
from collections import deque

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"  # Avoid memory leak on Windows

from sklearn.cluster import KMeans

# FUNCTIONS


# Shape function
def shape(xi, eta):
    """Shape functions for a 4-node, iso-parametric element..."""
    N = (
        np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )
        * 0.25
    )
    return N


# Gradient of shape functions
def gradShape(xi, eta):
    """Gradient of the shape functions..."""
    dN = (
        np.array(
            [
                [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],  # dN/dxi
                [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],  # dN/deta
            ]
        )
        * 0.25
    )
    return np.array(dN)


# PRE-PROCESS

# Mesh

nElex = 60
nEley = 20

nEle = nElex * nEley
nNode = (nElex + 1) * (nEley + 1)
nDof = 2 * nNode

# Material
Emax = 1.0e2  # GPa

x = np.genfromtxt("EOutput.txt")
x = np.where(x == 1e-5, 1e-15, 1)
E = Emax * np.ones((nEle, 1)) * x[:, np.newaxis]
nu = 0.3
# coeff = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
# DBase = np.array(
#     [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 - nu]])
DBase = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
coeff = E / (1.0 - nu**2)
D = np.array([coeff[i] * DBase for i in range(len(coeff))])

# Nodes coordinates
nodeCoordx = np.linspace(0, nElex, nElex + 1)
nodeCoordy = np.linspace(0, nEley, nEley + 1)
X, Y = np.meshgrid(nodeCoordx, nodeCoordy)
nodeCoord = np.column_stack((X.ravel(), Y.ravel()))

# Element
eTop = np.zeros((nElex * nEley, 4), dtype=int)
nodeOffsets = np.array(
    [
        0,
        1,
        nElex + 2,
        nElex + 1,
    ]
)

for eley in range(nEley):
    for elex in range(nElex):
        ele = elex + eley * nElex
        eTop[ele, :] = (elex + eley * (nElex + 1)) + nodeOffsets

# SOLVER

# Stiffness matrix
K = np.zeros((nDof, nDof))

# 2*2 Gauss Quadrature (4 Gauss points)
q4 = np.array(
    [
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
    ]
) / np.sqrt(3.0)

B = np.zeros((3, 8))

for index, e in enumerate(eTop):
    eleNodeCoord = nodeCoord[e, :]
    Ke = np.zeros((8, 8))
    for q in q4:
        dN = gradShape(q[0], q[1])
        J = np.dot(dN, eleNodeCoord).T
        dN = np.dot(np.linalg.inv(J), dN)

        B[0, 0::2] = dN[0, :]
        B[1, 1::2] = dN[1, :]
        B[2, 0::2] = dN[1, :]
        B[2, 1::2] = dN[0, :]

        Ke += np.dot(np.dot(B.T, D[index]), B) * np.linalg.det(J)

    for i, I in enumerate(e):
        for j, J in enumerate(e):
            K[2 * I, 2 * J] += Ke[2 * i, 2 * j]
            K[2 * I + 1, 2 * J] += Ke[2 * i + 1, 2 * j]
            K[2 * I + 1, 2 * J + 1] += Ke[2 * i + 1, 2 * j + 1]
            K[2 * I, 2 * J + 1] += Ke[2 * i, 2 * j + 1]

K = sp.csr_matrix(K)
K = K.tolil()

# Boundary conditions
f = np.zeros(nDof)
loadDof = 2 * np.where(np.all(np.isclose(nodeCoord, [0, nEley]), axis=1))[0] + 1
# loadDof = 2 * np.where(np.all(np.isclose(nodeCoord, [nElex, nEley / 2]), axis=1))[0] + 1
f[loadDof] = -1

leftDof = 2 * np.where(np.isclose(nodeCoord[:, 0], 0.0))[0]
hingeNode = np.where(np.all(np.isclose(nodeCoord, [nElex, 0.0]), axis=1))[0]
hingeDof = np.hstack([2 * hingeNode + 1])

fixedDof = np.hstack([leftDof, hingeDof])

# fixedNode = np.where(np.isclose(nodeCoord[:, 0], 0.0))[0]
# fixedDof = np.hstack([2 * fixedNode, 2 * fixedNode + 1])

for dof in fixedDof:
    K[dof, :] = 0.0
    K[:, dof] = 0.0
    K[dof, dof] = 1.0
    f[dof] = 0

K = K.tocsr()

u = spla.spsolve(K, f)

# POST-PROCESS

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True

# Stress
sigma = np.zeros((nNode, 3))

# Calculate stress at each node
for index, e in enumerate(eTop):
    eleNodeCoord = nodeCoord[e, :]
    eleDisp = np.array(
        [
            u[2 * e[0]],
            u[2 * e[0] + 1],
            u[2 * e[1]],
            u[2 * e[1] + 1],
            u[2 * e[2]],
            u[2 * e[2] + 1],
            u[2 * e[3]],
            u[2 * e[3] + 1],
        ]
    )

    for q in q4:
        dN = gradShape(q[0], q[1])
        J = np.dot(dN, eleNodeCoord).T
        dN = np.dot(np.linalg.inv(J), dN)

        B[0, 0::2] = dN[0, :]
        B[1, 1::2] = dN[1, :]
        B[2, 0::2] = dN[1, :]
        B[2, 1::2] = dN[0, :]

        # Calculate strain
        strain = np.dot(B, eleDisp)

        # Calculate stress
        stress = np.dot(D[index], strain)

        # Accumulate stress for each node in the element
        for i, node in enumerate(e):
            sigma[node] += stress

# Average stress at each node by dividing by number of elements connected
elementCounts = np.zeros(nNode)
for e in eTop:
    elementCounts[e] += 4

sigma /= elementCounts[:, np.newaxis]

vonMisesSig = np.sqrt(
    sigma[:, 0] ** 2
    - sigma[:, 0] * sigma[:, 1]
    + sigma[:, 1] ** 2
    + 3 * sigma[:, 2] ** 2
)

majorPriSig = (sigma[:, 0] + sigma[:, 1]) / 2 + np.sqrt(
    ((sigma[:, 0] - sigma[:, 1]) / 2) ** 2 + sigma[:, 2] ** 2
)

minorPriSig = (sigma[:, 0] + sigma[:, 1]) / 2 - np.sqrt(
    ((sigma[:, 0] - sigma[:, 1]) / 2) ** 2 + sigma[:, 2] ** 2
)

tauPriMax = (majorPriSig - minorPriSig) / 2

fig, ax = plt.subplots(figsize=(8, 4))
triAng = Triangulation(nodeCoord[:, 0], nodeCoord[:, 1])
contour = ax.tricontourf(
    triAng,
    vonMisesSig,
    levels=14,
    cmap="bone_r",
    vmin=0.1,
)

# fig.colorbar(contour)
ax.set_aspect("equal")
ax.yaxis.set_label_position("right")
# plt.ylabel("Von. Mises Stress")
plt.xlim(0 - 5, nElex + 5)
plt.xticks = np.arange(0 - 5, nElex + 5, 10)
plt.ylim(0 - 5, nEley + 5)
plt.yticks = np.arange(0 - 5, nEley + 5, 20)
# plt.grid()

# Displacement
disp = np.sqrt(u[0::2] ** 2 + u[1::2] ** 2)

dispCoord = nodeCoord + np.column_stack((u[0::2], u[1::2]))

nodeX = np.reshape(dispCoord[:, 0], [nEley + 1, nElex + 1])
nodeY = np.reshape(dispCoord[:, 1], [nEley + 1, nElex + 1])
dispZ = np.reshape(disp, [nEley + 1, nElex + 1])

plt.figure(figsize=(8, 4))

mask = np.ones((nElex + 1) * (nEley + 1), dtype=bool)

for index, e in enumerate(eTop):
    plt.plot(
        np.append(nodeCoord[e, 0], nodeCoord[e[0], 0]),
        np.append(nodeCoord[e, 1], nodeCoord[e[0], 1]),
        "b:",
        alpha=0.1,
    )
    if x[index] == 1:
        plt.plot(
            np.append(dispCoord[e, 0], dispCoord[e[0], 0]),
            np.append(dispCoord[e, 1], dispCoord[e[0], 1]),
            "w-",
            alpha=0.25,
        )
        mask[e] = False
mask = np.reshape(mask, [nEley + 1, nElex + 1])
plt.contourf(nodeX, nodeY, np.ma.masked_array(dispZ, mask=mask), levels=14, cmap="GnBu")
# cbar = plt.colorbar()
# cbar.set_label(r"Displacement ($d=\sqrt{u^2+v^2}$)")

plt.xlim(0 - 5, nElex + 5)
plt.xticks = np.arange(0 - 5, nElex + 5, 10)
plt.ylim(0 - 5, nEley + 5)
plt.yticks = np.arange(0 - 5, nEley + 5, 10)
plt.axis("equal")
# plt.show()

# Principal stress tensor glyphs
sigmaPS = np.zeros((nNode, 2))
sigmaPSAngleMajor = np.zeros(nNode)
sigmaPSAngleMinor = np.zeros(nNode)

sigmaXX = sigma[:, 0]
sigmaYY = sigma[:, 1]
tauXY = sigma[:, 2]

for n in range(nNode):
    stressTensor = np.array([[sigmaXX[n], tauXY[n]], [tauXY[n], sigmaYY[n]]])
    eigVals, eigVecs = np.linalg.eigh(stressTensor)

    sigmaPS[n, 0] = eigVals[1]  # major principal stress
    sigmaPS[n, 1] = eigVals[0]  # minor principal stress

    sigmaPSAngleMajor[n] = np.arctan2(eigVecs[1, 1], eigVecs[0, 1])
    sigmaPSAngleMinor[n] = np.arctan2(eigVecs[1, 0], eigVecs[0, 0])

fig, ax = plt.subplots(figsize=(8, 4))

for n in range(nNode):

    glyphs = Ellipse(
        xy=(nodeCoord[n, 0], nodeCoord[n, 1]),
        width=abs(sigmaPS[n, 0]) * 2,
        height=abs(sigmaPS[n, 1]) * 2,
        angle=np.degrees(sigmaPSAngleMajor[n]),
        edgecolor="grey",
        facecolor="none",
        lw=1.5,
    )
    ax.add_patch(glyphs)

    # Minor principal stress axis
    minorAxisL = sigmaPS[n, 1]
    minorX1 = nodeCoord[n, :] + minorAxisL * np.array(
        [
            np.sin(sigmaPSAngleMajor[n]),
            -np.cos(sigmaPSAngleMajor[n]),
        ]
    )
    minorX2 = nodeCoord[n, :] - minorAxisL * np.array(
        [
            np.sin(sigmaPSAngleMajor[n]),
            -np.cos(sigmaPSAngleMajor[n]),
        ]
    )
    if sigmaPS[n, 1] >= 0:
        ax.plot(
            [minorX1[0], minorX2[0]],
            [minorX1[1], minorX2[1]],
            color="forestgreen",
            lw=1,
        )
    else:
        ax.plot([minorX1[0], minorX2[0]], [minorX1[1], minorX2[1]], color="wheat", lw=1)

    # Major principal stress axis
    majorAxisL = sigmaPS[n, 0]
    majorX1 = nodeCoord[n, :] + majorAxisL * np.array(
        [
            np.cos(sigmaPSAngleMajor[n]),
            np.sin(sigmaPSAngleMajor[n]),
        ]
    )
    majorX2 = nodeCoord[n, :] - majorAxisL * np.array(
        [
            np.cos(sigmaPSAngleMajor[n]),
            np.sin(sigmaPSAngleMajor[n]),
        ]
    )
    if sigmaPS[n, 0] >= 0:
        ax.plot(
            [majorX1[0], majorX2[0]], [majorX1[1], majorX2[1]], color="royalblue", lw=1
        )
    else:
        ax.plot([majorX1[0], majorX2[0]], [majorX1[1], majorX2[1]], color="coral", lw=1)

ax.set_aspect("equal")
plt.xlim(0 - 5, nElex + 5)
plt.ylim(0 - 5, nEley + 5)
plt.grid()

# Principal stress trajectory


def eleIndex(nodeCoord, eTop, x, y):
    """Find the element that contains the point (x, y)"""
    for ele in range(len(eTop)):
        eleNodes = eTop[ele]
        eleCoords = nodeCoord[eleNodes]

        minX = np.min(eleCoords[:, 0])
        maxX = np.max(eleCoords[:, 0])
        minY = np.min(eleCoords[:, 1])
        maxY = np.max(eleCoords[:, 1])

        if minX <= x <= maxX and minY <= y <= maxY:
            return ele
    return None


def calPS(x, y, sigmaPS, isMinor, nodeCoord, eTop):
    eleId = eleIndex(nodeCoord, eTop, x, y)
    eleNodes = eTop[eleId]
    nodeCoords = nodeCoord[eleNodes]

    majorPSs = sigmaPS[eleNodes, 0]
    minorPSs = sigmaPS[eleNodes, 1]
    # thetas = sigmaPSAngle[eleNodes]
    sigmaXXs = sigma[eleNodes, 0]
    sigmaYYs = sigma[eleNodes, 1]
    taus = sigma[eleNodes, 2]

    x1, y1 = nodeCoords[0]
    x2, y2 = nodeCoords[1]
    x3, y3 = nodeCoords[2]
    x4, y4 = nodeCoords[3]

    xi = (x - x1) / (x2 - x1) + (x - x4) / (x3 - x4) - 1
    eta = (y - y1) / (y3 - y1) + (y - y2) / (y4 - y2) - 1

    N = shape(xi, eta)

    majorPS = np.dot(N, majorPSs)
    minorPS = np.dot(N, minorPSs)
    sigmaXX = np.dot(N, sigmaXXs)
    sigmaYY = np.dot(N, sigmaYYs)
    tau = np.dot(N, taus)

    stressTensor = np.array([[sigmaXX, tau], [tau, sigmaYY]])
    eigVals, eigVecs = np.linalg.eigh(stressTensor)
    if isMinor:
        theta = np.arctan2(eigVecs[1, 0], eigVecs[0, 0])
        vector = eigVecs[:, 0]
    else:
        theta = np.arctan2(eigVecs[1, 1], eigVecs[0, 1])
        vector = eigVecs[:, 1]

    return eleId, majorPS, minorPS, theta, vector


def genTraj(points, isMinor=True, stepSize=0.25, maxNSteps=1000):
    trajectory = []
    trajLineStr = []

    for point in points:
        x, y = point
        traceX = [x]
        traceY = [y]
        thetaOld = None
        vectorOld = None

        for direction in [-1, 1]:
            x, y = point
            for n in range(maxNSteps):
                eleId, majorPS, minorPS, theta, vector = calPS(
                    x,
                    y,
                    sigmaPS,
                    isMinor,
                    nodeCoord,
                    eTop,
                )

                if vectorOld is not None and np.dot(vector, vectorOld) < 0:
                    # theta = theta - pi
                    vector = -vector

                dx = direction * stepSize * vector[0]
                dy = direction * stepSize * vector[1]

                x += dx
                y += dy

                if E[eleId] <= 1e-3 or not (0 <= x <= nElex and 0 <= y <= nEley):
                    break

                thetaOld = theta
                vectorOld = vector

                if direction == 1:
                    traceX.append(x)
                    traceY.append(y)
                    # pass
                else:
                    traceX.insert(0, x)
                    traceY.insert(0, y)

            if len(traceX) >= 2 and len(traceY) >= 2:
                trajLineStr.append(LineString(zip(traceX, traceY)))
            trajectory.append((traceX, traceY))

    return trajectory, trajLineStr


fig, ax = plt.subplots(figsize=(8, 4))
for index, e in enumerate(eTop):
    xCoords = np.append(nodeCoord[e, 0], nodeCoord[e[0], 0])
    yCoords = np.append(nodeCoord[e, 1], nodeCoord[e[0], 1])
    plt.plot(
        xCoords,
        yCoords,
        "b:",
        alpha=0.1,
    )

# Start points of principal stress trajectories
xCoords = 0.0
yCoords = np.linspace(0, nEley, 20)
startPoints = []
# for i in xCoords:
#     for j in yCoords:
#         eleId = eleIndex(nodeCoord, eTop, i, j)

#         if eleId is not None and x[eleId] == 1:
#             startPoints.append([i, j])

for j in yCoords:
    i = xCoords
    eleId = eleIndex(nodeCoord, eTop, i, j)

    if eleId is not None and x[eleId] == 1:
        startPoints.append([i, j])

# for p in startPoints:
#     plt.plot(
#         p[0],
#         p[1],
#         marker="o",
#         markerfacecolor="w",
#         markeredgecolor="royalblue",
#     )

# Minor principal stress trajectories
minorPSStart = startPoints
minorTrajectories, minorTrajLineStr = genTraj(minorPSStart, isMinor=True)
for traceX, traceY in minorTrajectories:
    plt.plot(traceX, traceY, color="coral", lw=2)

# Major principal stress trajectories
majorPSStart = startPoints
majorTrajectories, majorTrajLineStr = genTraj(majorPSStart, isMinor=False)
for traceX, traceY in majorTrajectories:
    plt.plot(traceX, traceY, color="royalblue", lw=2)

ax.set_aspect("equal")
plt.xlim(0 - 5, nElex + 5)
plt.ylim(0 - 5, nEley + 5)
plt.grid()

# Intersection points of principal stress trajectories

intsectPoints = []
# Intersection points
for minor in minorTrajLineStr:
    for major in majorTrajLineStr:
        intsectPoints.append(minor.intersection(major))

alterPoints = startPoints
for p in intsectPoints:
    if p.geom_type == "Point":
        startPoints.append([p.x, p.y])
alterPoints = [
    list(coord) for coord in set(tuple(point) for point in alterPoints)
]  # unique points

alterPoints = np.array(alterPoints)

nCluster = 5
kMeans = KMeans(n_clusters=nCluster)
kMeans.fit(np.array(alterPoints))
clusterLabels = kMeans.labels_


fig, ax = plt.subplots(figsize=(8, 4))
for index, e in enumerate(eTop):
    xCoords = np.append(nodeCoord[e, 0], nodeCoord[e[0], 0])
    yCoords = np.append(nodeCoord[e, 1], nodeCoord[e[0], 1])
    plt.plot(
        xCoords,
        yCoords,
        "b:",
        alpha=0.1,
    )
    if x[index] == 1:
        plt.fill(
            xCoords,
            yCoords,
            color="silver",
            alpha=0.5,
        )

plt.scatter(
    alterPoints[:, 0],
    alterPoints[:, 1],
    marker="D",
    c=clusterLabels,
    cmap="Paired",
    edgecolors="white",
)

ax.set_aspect("equal")
plt.xlim(0 - 5, nElex + 5)
plt.ylim(0 - 5, nEley + 5)
plt.grid()

# Selected points as nodes

meanCoords = kMeans.cluster_centers_

fig, ax = plt.subplots(figsize=(8, 4))
for index, e in enumerate(eTop):
    xCoords = np.append(nodeCoord[e, 0], nodeCoord[e[0], 0])
    yCoords = np.append(nodeCoord[e, 1], nodeCoord[e[0], 1])
    plt.plot(
        xCoords,
        yCoords,
        "b:",
        alpha=0.1,
    )
    if x[index] == 1:
        plt.fill(
            xCoords,
            yCoords,
            color="silver",
            alpha=0.5,
        )
plt.scatter(
    meanCoords[:, 0],
    meanCoords[:, 1],
    marker="D",
    edgecolors="white",
)

ax.set_aspect("equal")
plt.xlim(0 - 5, nElex + 5)
plt.ylim(0 - 5, nEley + 5)
plt.grid()

nodeOutput = meanCoords
np.savetxt("nodeOutput.txt", nodeOutput)

plt.show()
