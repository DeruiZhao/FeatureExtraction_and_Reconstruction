import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.ndimage import convolve
from collections import defaultdict

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Mesh parameters
nElex = 60
nEley = 20

# Load and preprocess data
x = np.genfromtxt("EOutput.txt")
x = np.where(x == 1e-5, 0, 1)
mesh = x.reshape((nEley, nElex))

# Load node coordinates and convert to element indices
meanCoords = np.genfromtxt("nodeOutput.txt")
sortIndex = np.lexsort((meanCoords[:, 1], meanCoords[:, 0]))
meanCoords = meanCoords[sortIndex]
nodeEle = [(int(coord[1]), int(coord[0])) for coord in meanCoords]  # (row, col) format
numNodes = len(nodeEle)


def is_valid(x, y, mesh):
    """Check if the position is valid and contains material"""
    return 0 <= x < mesh.shape[0] and 0 <= y < mesh.shape[1] and mesh[x, y] == 1


def neighbors(x, y):
    """Get valid neighboring positions"""
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(x + dx, y + dy) for dx, dy in offsets if is_valid(x + dx, y + dy, mesh)]


def plotExpansionMap(expansionMap, iteration):
    """Plot the current expansion map"""
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # ax1 = axes[0]
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(
        expansionMap,
        cmap="bone_r",
        origin="upper",
        extent=[0, nElex, nEley, 0],
    )

    # Plot nodes
    for node in nodeEle:
        ax1.plot(
            node[1],
            node[0],
            marker="o",
            color="wheat",
            markeredgecolor="white",
        )
        ax1.plot(
            node[1],
            node[0],
            marker="o",
            color="none",
            markeredgecolor="white",
            markersize=12,
        )
        ax1.plot(
            node[1],
            node[0],
            marker="o",
            color="none",
            markeredgecolor="white",
            markersize=16,
        )

    ax1.set_aspect("equal")
    plt.xlim(-5, nElex + 5)
    plt.ylim(-5, nEley + 5)
    # plt.show()

    # ax2 = axes[1]
    ax2 = fig.add_subplot(gs[1])
    mask = np.zeros_like(topMat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask = (np.flipud(mask) - 1) * (-1)
    mask = np.rot90(mask, k=1)

    sns.heatmap(
        topMat,
        annot=True,
        cmap="bone_r",
        cbar=False,
        square=True,
        xticklabels=np.arange(1, numNodes + 1),
        yticklabels=np.arange(1, numNodes + 1),
        linewidths=4,
        linecolor="white",
        mask=mask,
        ax=ax2,
    )

    plt.tight_layout()
    plt.show()


# Initialize topology matrix and expansion map
topMat = np.zeros((numNodes, numNodes), dtype=int)
expansionMap = np.full(mesh.shape, -1, dtype=int)

# Initialize queue with all starting points
queue = deque()
for idx, (x, y) in enumerate(nodeEle):
    queue.append((x, y, idx))  # (x, y, source_index)
    expansionMap[x, y] = idx

# Expand from all points simultaneously
iteration = 0
while queue:
    currentX, currentY, sourceIdx = queue.popleft()

    # Check if current position contains another node
    for targetIdx, (targetX, targetY) in enumerate(nodeEle):
        if targetIdx != sourceIdx and (targetX, targetY) == (currentX, currentY):
            topMat[sourceIdx, targetIdx] = 1
            topMat[targetIdx, sourceIdx] = 1

    # Explore neighbors
    for nextX, nextY in neighbors(currentX, currentY):
        if expansionMap[nextX, nextY] == -1:
            # Unexplored position
            expansionMap[nextX, nextY] = sourceIdx
            queue.append((nextX, nextY, sourceIdx))
        elif expansionMap[nextX, nextY] != sourceIdx:
            # Position reached by another node
            otherIdx = expansionMap[nextX, nextY]
            topMat[sourceIdx, otherIdx] = 1
            topMat[otherIdx, sourceIdx] = 1

# Intersection boundaries between regions

# Define the 4-connected convolution kernel
kernel = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)

uniqueLabels = np.arange(numNodes)  # Extract all unique region labels
boundaryPixels = []

for label in uniqueLabels:
    # Create a binary mask for the current region
    binaryMask = (expansionMap == label).astype(int)

    # Apply convolution to detect boundaries between the current region and others
    boundaryMap = convolve(binaryMask, kernel, mode="constant", cval=0)

    # Find boundary pixels that are adjacent to pixels from a different region
    boundaryIndices = np.argwhere((boundaryMap > 0) & (expansionMap != label))

    # Record the boundary pixels and their neighboring regions
    for idx in boundaryIndices:
        i, j = idx
        # Get the label of the current pixel and its neighboring region label
        currentLabel = label
        neighborLabel = expansionMap[i, j]
        if neighborLabel >= 0:
            boundaryPixels.append((i, j, currentLabel, neighborLabel))

boundaryImage = np.zeros_like(expansionMap, dtype=np.uint8)
for i, j, _, _ in boundaryPixels:
    boundaryImage[i, j] = 1

# Member width i.e. boundary length
boundaryCounts = defaultdict(int)
for i, j, label1, label2 in boundaryPixels:
    regionPair = tuple(sorted([label1, label2]))
    boundaryCounts[regionPair] += 1

memberWidthMat = np.zeros_like(topMat, dtype=float)
for (label1, label2), count in boundaryCounts.items():
    memberWidthMat[label1, label2] = count / 2
    memberWidthMat[label2, label1] = count / 2

#  Figure initialisation
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True

# Visualization
# Nodes positions
fig, ax = plt.subplots(figsize=(8, 4))
plt.imshow(
    mesh,
    cmap="bone_r",
    vmin=0,
    vmax=3,
    origin="upper",
    extent=[0, nElex, nEley, 0],
)

# plt.scatter(
#     meanCoords[:, 0],
#     meanCoords[:, 1],
#     marker="D",
#     color="wheat",
#     edgecolors="white",
#     s=160,
#     zorder=2,
# )

for i, (x, y) in enumerate(meanCoords):
    plt.plot(
        x,
        y,
        marker="D",
        color="royalblue",
        markeredgecolor="white",
        markersize=16,
        zorder=2,
    )
    plt.annotate(
        str(i + 1),
        (x, y),
        color="white",
        ha="center",
        va="center",
        fontweight=700,
    )
    # plt.text(
    #     x,
    #     y,
    #     str(i + 1),
    #     color="white",
    #     ha="center",
    #     va="center",
    #     fontweight=700,
    # )

ax.set_aspect("equal")
plt.xlim(-5, nElex + 5)
plt.ylim(-5, nEley + 5)
plt.grid()


# Plot expansion map
fig, ax = plt.subplots(figsize=(8, 4))
plt.imshow(
    expansionMap,
    cmap="bone_r",
    origin="upper",
    extent=[0, nElex, nEley, 0],
)

# Plot nodes
for node in nodeEle:
    plt.plot(
        node[1],
        node[0],
        marker="o",
        color="wheat",
        markeredgecolor="white",
    )
    plt.plot(
        node[1],
        node[0],
        marker="o",
        color="none",
        markeredgecolor="white",
        markersize=12,
    )
    plt.plot(
        node[1],
        node[0],
        marker="o",
        color="none",
        markeredgecolor="white",
        markersize=16,
    )

ax.set_aspect("equal")
plt.xlim(-5, nElex + 5)
plt.ylim(-5, nEley + 5)
# plt.grid()


# Topology matrix of nodes
topMatUpper = np.triu(topMat)

fig, ax = plt.subplots(figsize=(8, 4))
# plt.imshow(
#     topMat,
#     cmap="bone_r",
#     interpolation="nearest",
# )

# plt.xticks(ticks=np.arange(0, numNodes), labels=np.arange(1, numNodes + 1))
# plt.yticks(ticks=np.arange(0, numNodes), labels=np.arange(1, numNodes + 1))
mask = np.zeros_like(topMat, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# mask2 = mask
mask = (np.flipud(mask) - 1) * (-1)
mask = np.rot90(mask, k=1)

sns.heatmap(
    topMat,
    annot=True,
    cmap="bone_r",
    cbar=False,
    square=True,
    xticklabels=np.arange(1, numNodes + 1),
    yticklabels=np.arange(1, numNodes + 1),
    linewidths=4,
    linecolor="white",
    mask=mask,
    ax=ax,
)

# Truss skeleton
fig, ax = plt.subplots(figsize=(8, 4))
plt.imshow(
    mesh,
    cmap="bone_r",
    vmin=0,
    vmax=3,
    origin="upper",
    extent=[0, nElex, nEley, 0],
)


# Draw connections between nodes using topMat
for i in range(numNodes):
    for j in range(i + 1, numNodes):  # Avoid drawing duplicate lines
        if topMat[i, j]:
            plt.plot(
                [meanCoords[i][0], meanCoords[j][0]],  # x coordinates
                [meanCoords[i][1], meanCoords[j][1]],  # y coordinates
                color="royalblue",
                linewidth=4,
                alpha=1,
            )

plt.scatter(
    meanCoords[:, 0],
    meanCoords[:, 1],
    marker="D",
    color="wheat",
    edgecolors="white",
    zorder=2,
)

ax.set_aspect("equal")
plt.xlim(-5, nElex + 5)
plt.ylim(-5, nEley + 5)
plt.grid()

# Intersection boundaries
fig, ax = plt.subplots(figsize=(8, 4))
plt.imshow(
    expansionMap,
    cmap="bone_r",
    origin="upper",
    extent=[0, nElex, nEley, 0],
)

maskBoundary = np.ma.masked_where(boundaryImage == 0, boundaryImage)
plt.imshow(
    maskBoundary,
    # cmap="binary",
    cmap=plt.matplotlib.colors.ListedColormap(["red"]),
    # alpha=0.5,
    extent=[0, nElex, nEley, 0],
)

ax.set_aspect("equal")
plt.xlim(-5, nElex + 5)
plt.ylim(-5, nEley + 5)
# plt.grid()


# Truss
fig, ax = plt.subplots(figsize=(8, 4))
# plt.imshow(
#     mesh,
#     cmap="bone_r",
#     vmin=0,
#     vmax=3,
#     origin="upper",
#     extent=[0, nElex, nEley, 0],
# )

# Draw connections between nodes using memberWidthMat
for i in range(numNodes):
    for j in range(i + 1, numNodes):  # Avoid drawing duplicate lines
        if topMat[i, j]:
            lineWidth = memberWidthMat[i, j] * 8
            plt.plot(
                [meanCoords[i][0], meanCoords[j][0]],  # x coordinates
                [meanCoords[i][1], meanCoords[j][1]],  # y coordinates
                color="royalblue",
                linewidth=lineWidth,
                solid_capstyle="butt",
                alpha=0.8,
            )

plt.scatter(
    meanCoords[:, 0],
    meanCoords[:, 1],
    marker="o",
    color="royalblue",
    edgecolors="white",
    zorder=2,
)

ax.set_aspect("equal")
plt.xlim(-5, nElex + 5)
plt.ylim(-5, nEley + 5)
# plt.grid()

plt.show()
