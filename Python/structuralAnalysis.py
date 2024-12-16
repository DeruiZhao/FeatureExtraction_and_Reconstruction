import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# PRE-PROCESSING

E = 1e2

# Nodes coordinates
nodeCoord = np.array(
    [
        [-58.4727, 2.55103],
        [-33.4632, 17.5214],
        [-15.1378, 2.86231],
        [0, 18.4901],
        [15.1378, 2.86231],
        [33.4632, 17.5214],
        [58.4727, 2.55103],
    ]
)

# Element top
eTop = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [3, 4],
        [3, 5],
        [4, 5],
        [4, 6],
        [5, 6],
    ]
)

nCoord = nodeCoord.shape[0]
nEle = eTop.shape[0]
nDof = 2

# A = 3 * np.ones(nEle)
A = [
    2.5,
    3,
    2,
    5,
    2,
    5,
    2,
    5,
    2,
    3,
    2.5,
]

# Boundary conditions
supports = np.zeros((nCoord, nDof))
supports[0, 0:2] = 1
supports[6, 1] = 1

forces = np.zeros((nCoord, nDof))
forces[3, 1] = -2

# SOLVER

eleLength = np.zeros(nEle)
cosine = np.zeros((nEle, 2))
for i in range(nEle):
    dx = nodeCoord[eTop[i, 1], 0] - nodeCoord[eTop[i, 0], 0]
    dy = nodeCoord[eTop[i, 1], 1] - nodeCoord[eTop[i, 0], 1]
    eleLength[i] = np.sqrt(dx**2 + dy**2)

    cosine[i, 0] = dx / eleLength[i]
    cosine[i, 1] = dy / eleLength[i]

K = np.zeros((nCoord * nDof, nCoord * nDof))

for i in range(nEle):
    # Element stiffness matrix
    c, s = cosine[i, 0], cosine[i, 1]
    KE = (
        E
        * A[i]
        / eleLength[i]
        * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ]
        )
    )

    eleDof = [
        2 * eTop[i, 0],
        2 * eTop[i, 0] + 1,
        2 * eTop[i, 1],
        2 * eTop[i, 1] + 1,
    ]

    # Assembly
    for row in range(4):
        for col in range(4):
            K[eleDof[row], eleDof[col]] += KE[row, col]

# Apply boundary conditions (penalty method)
penalty = 1e15
for i in range(nCoord):
    for j in range(nDof):
        if supports[i, j] == 1:
            K[2 * i + j, 2 * i + j] += penalty

load = forces.flatten()
U = np.linalg.solve(K, load)
U = U.reshape((nCoord, nDof))

# Stress calculation
stress = np.zeros(nEle)
for i in range(nEle):
    dx = U[eTop[i, 1], 0] - U[eTop[i, 0], 0]
    dy = U[eTop[i, 1], 1] - U[eTop[i, 0], 1]

    axialStrain = (cosine[i, 0] * dx + cosine[i, 1] * dy) / eleLength[i]
    stress[i] = E * axialStrain

# VISUALIZATION
#  Figure initialisation
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True

# Original structure
fig, ax = plt.subplots(figsize=(12, 3.2))

for i in range(nEle):
    node1 = eTop[i, 0]
    node2 = eTop[i, 1]
    plt.plot(
        [nodeCoord[node1, 0], nodeCoord[node2, 0]],
        [nodeCoord[node1, 1], nodeCoord[node2, 1]],
        color="royalblue",
        linewidth=4,
        solid_capstyle="butt",
    )
    plt.plot(
        (nodeCoord[node1, 0] + nodeCoord[node2, 0]) / 2,
        (nodeCoord[node1, 1] + nodeCoord[node2, 1]) / 2,
        marker="D",
        color="white",
        markeredgecolor="silver",
        markersize=24,
    )
    plt.annotate(
        str(i + 1),
        (
            (nodeCoord[node1, 0] + nodeCoord[node2, 0]) / 2,
            (nodeCoord[node1, 1] + nodeCoord[node2, 1]) / 2 - 0.2,
        ),
        color="black",
        ha="center",
        va="center",
        fontweight=700,
    )


supportedNode = np.where(np.any(supports == 1, axis=1))[0]
for node in supportedNode:
    x, y = nodeCoord[node]
    directions = supports[node]

    # Check x-direction support
    if directions[0] == 1:
        plt.plot(
            x - 1,
            y,
            marker=">",
            color="navy",
            markeredgecolor="black",
            markersize=10,
        )

    # Check y-direction support
    if directions[1] == 1:
        plt.plot(
            x,
            y - 1,
            marker="^",
            color="navy",
            markeredgecolor="black",
            markersize=10,
        )

forceNode = np.where(np.any(forces != 0, axis=1))[0]
for node in forceNode:
    x, y = nodeCoord[node]
    directions = forces[node]

    plt.arrow(
        x,
        y,
        directions[0] * 5,
        directions[1] * 5,
        width=0.5,
        facecolor="crimson",
        edgecolor="white",
    )

plt.scatter(
    nodeCoord[:, 0],
    nodeCoord[:, 1],
    marker="o",
    color="royalblue",
    edgecolors="white",
    zorder=2,
)

ax.set_aspect("equal")
plt.xlim(-65, 60 + 5)
plt.ylim(-5, 20 + 5)
plt.grid()

# Original structure with shape
fig, ax = plt.subplots(figsize=(12, 4))

for i in range(nEle):
    node1 = eTop[i, 0]
    node2 = eTop[i, 1]
    plt.plot(
        [nodeCoord[node1, 0], nodeCoord[node2, 0]],
        [nodeCoord[node1, 1], nodeCoord[node2, 1]],
        color="royalblue",
        linewidth=A[i] * 8,
        solid_capstyle="butt",
    )

supportedNode = np.where(np.any(supports == 1, axis=1))[0]
for node in supportedNode:
    x, y = nodeCoord[node]
    directions = supports[node]

    # Check x-direction support
    if directions[0] == 1:
        plt.plot(
            x - 1,
            y,
            marker=">",
            color="navy",
            markeredgecolor="black",
            markersize=10,
        )

    # Check y-direction support
    if directions[1] == 1:
        plt.plot(
            x,
            y - 1,
            marker="^",
            color="navy",
            markeredgecolor="black",
            markersize=10,
        )

plt.scatter(
    nodeCoord[:, 0],
    nodeCoord[:, 1],
    marker="o",
    color="royalblue",
    edgecolors="white",
    zorder=2,
)

ax.set_aspect("equal")
plt.xlim(-65, 60 + 5)
plt.ylim(-5, 20 + 5)
plt.grid()

# Stress and displacement
fig, ax = plt.subplots(figsize=(12, 3))

norm = mcolors.Normalize(vmin=stress.min(), vmax=stress.max())

sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
sm.set_array([])

for i in range(nEle):
    node1 = eTop[i, 0]
    node2 = eTop[i, 1]
    x1 = nodeCoord[node1, 0] + U[node1, 0]
    y1 = nodeCoord[node1, 1] + U[node1, 1]
    x2 = nodeCoord[node2, 0] + U[node2, 0]
    y2 = nodeCoord[node2, 1] + U[node2, 1]
    # color = plt.cm.RdBu((stress[i] - stress.min()) / (stress.max() - stress.min()))
    color = sm.to_rgba(stress[i])
    plt.plot(
        # [nodeCoord[node1, 0], nodeCoord[node2, 0]],
        # [nodeCoord[node1, 1], nodeCoord[node2, 1]],
        [x1, x2],
        [y1, y2],
        color=color,
        linewidth=A[i] * 8,
        solid_capstyle="butt",
    )
fig.colorbar(
    sm,
    ax=ax,
    label="Stress ($GPa$)",
)

supportedNode = np.where(np.any(supports == 1, axis=1))[0]
for node in supportedNode:
    x, y = nodeCoord[node] + U[node]
    directions = supports[node]

    # Check x-direction support
    if directions[0] == 1:
        plt.plot(
            x - 1,
            y,
            marker=">",
            color="navy",
            markeredgecolor="black",
            markersize=10,
        )

    # Check y-direction support
    if directions[1] == 1:
        plt.plot(
            x,
            y - 1,
            marker="^",
            color="navy",
            markeredgecolor="black",
            markersize=10,
        )

plt.scatter(
    nodeCoord[:, 0] + U[:, 0],
    nodeCoord[:, 1] + U[:, 1],
    marker="o",
    color="royalblue",
    edgecolors="white",
    zorder=2,
)

ax.set_aspect("equal")
plt.xlim(-65, 60 + 5)
plt.ylim(-5, 20 + 5)
# plt.grid()

plt.show()
