import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import colors
import cvxopt
import cvxopt.cholmod
from skimage import morphology


def KEMat():
    """Create an 8x8 stiffness matrix (KE) for a finite element under plane stress conditions with unit Young's modulus (E = 1) and Poisson's ratio (nu = 0.3)"""
    E = 1
    nu = 0.3
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
    return KE


def deleterowcol(A, delrow, delcol):
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A


# Pre-configuration
nElex = 60
nEley = 20
volFrac = 0.45
evoVolRatio = 0.02
rMin = 2
penal = 3
dMin = 2

print("Minimum Compliance Problem")
print("Elements: " + str(nElex) + "×" + str(nEley))
print(
    "volFrac: "
    + str(volFrac)
    + ", er: "
    + str(evoVolRatio)
    + ", rMin: "
    + str(rMin)
    + ", penal: "
    + str(penal)
)
print("Filter method: Sensitivity based")

# Young's modulus
E0 = 1.0

# DoFs
nDof = 2 * (nElex + 1) * (nEley + 1)

# Variable initialisation
x = np.ones(nElex * nEley, dtype=float)

# Nodes coordinates
nodeCoordx = np.linspace(0, nElex, nElex + 1)
nodeCoordy = np.linspace(0, nEley, nEley + 1)
X, Y = np.meshgrid(nodeCoordx, nodeCoordy)
nodeCoord = np.column_stack((X.ravel(), Y.ravel()))

eleCentroid = np.zeros((nElex * nEley, 2), dtype=float)

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

        eleCentroid[ele, 0] = nodeCoord[eTop[ele, :], 0].mean()
        eleCentroid[ele, 1] = nodeCoord[eTop[ele, :], 1].mean()

# Assemble stiffness matrix
KE = KEMat()  # Element stiffness matrix
gDof = np.zeros((nElex * nEley, 8), dtype=int)  # Global DoF matrix
twoElex = 2 * (nElex + 1)
dofOffsets = np.array(
    [
        0,
        1,
        2,
        3,
        twoElex + 2,
        twoElex + 3,
        twoElex,
        twoElex + 1,
    ]
)
for eley in range(nEley):
    for elex in range(nElex):
        node = elex + eley * (nElex + 1)
        ele = elex + eley * nElex
        gDof[ele, :] = 2 * node + dofOffsets

# Index of sparse matrix
iK = np.kron(gDof, np.ones((8, 1))).flatten()
jK = np.kron(gDof, np.ones((1, 8))).flatten()

#  Filter
nFilter = int(nElex * nEley * ((2 * (np.ceil(rMin) - 1) + 1) ** 2))
iH = np.ones(nFilter)
jH = np.ones(nFilter)
sH = np.zeros(nFilter)
cc = 0
for j in range(nEley):
    for loop in range(nElex):
        row = j * nElex + loop
        kk1 = int(np.maximum(loop - (np.ceil(rMin) - 1), 0))
        kk2 = int(np.minimum(loop + np.ceil(rMin), nElex))
        ll1 = int(np.maximum(j - (np.ceil(rMin) - 1), 0))
        ll2 = int(np.minimum(j + np.ceil(rMin), nEley))
        for l in range(ll1, ll2):
            for k in range(kk1, kk2):
                col = l * nElex + k
                fac = rMin - np.sqrt((loop - k) * (loop - k) + (j - l) * (j - l))
                iH[cc] = row
                jH[cc] = col
                sH[cc] = np.maximum(0.0, fac)
                cc = cc + 1

#  Filtering matrix
H = coo_matrix((sH, (iH, jH)), shape=(nElex * nEley, nElex * nEley)).tocsc()
Hs = H.sum(1)

# Boudary condition
dofs = np.arange(2 * (nElex + 1) * (nEley + 1))
fixedDofs = np.union1d(dofs[0 :: 2 * (nElex + 1)], np.array([2 * nElex + 1]))
# fixedDofs = np.union1d(dofs[0 :: 2 * (nElex + 1)], dofs[1 :: 2 * (nElex + 1)])
freeDofs = np.setdiff1d(dofs, fixedDofs)

f = np.zeros((nDof, 1))
u = np.zeros((nDof, 1))
f[2 * nEley * (nElex + 1) + 1, 0] = -1
# f[int(2 * (nEley / 2 * (nElex + 1) + nElex)) + 1, 0] = -1

#  Figure initialisation
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True

plt.show(block=True)
fig, ax = plt.subplots(figsize=(8, 4))  # figsize=(8, 3.5)
im = ax.imshow(
    x.reshape((nEley, nElex)),
    cmap="bone_r",
    # cmap="Blues",
    interpolation="none",
    norm=colors.Normalize(vmin=0, vmax=1),
    extent=[0, nElex, nEley, 0],
)
ax.set_xlim(0 - 5, nElex + 5)
ax.set_xticks = np.arange(0 - 5, nElex + 5, 10)
ax.set_ylim(0 - 5, nEley + 5)
ax.set_yticks = np.arange(nEley + 5, 0 - 5, 10)
# fig.colorbar(im, ax=ax)
fig.show()

# Iteration
vol = 1
loop = 0
change = 1

dc = np.ones(nEley * nElex)  # Design variable's compliance sensitivity
olddc = dc.copy()
ce = np.ones(nEley * nElex)  # Element strain energy
obj = []

while change > 0.0001 and loop < 1000:
    loop += 1

    vol = max(vol * (1 - evoVolRatio), volFrac)

    if loop > 1:
        olddc = dc

    # FEA
    sK = ((KE.flatten()[np.newaxis]).T * ((x) ** penal)).flatten(order="F")
    K = coo_matrix((sK, (jK, iK)), shape=(nDof, nDof)).tocsc()

    K = deleterowcol(K, fixedDofs, fixedDofs).tocoo()

    K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
    B = cvxopt.matrix(f[freeDofs, 0])
    cvxopt.cholmod.linsolve(K, B)
    u[freeDofs, 0] = np.array(B)[:, 0]

    # Objective function and sensitivity
    ce[:] = (
        np.dot(u[gDof].reshape(nElex * nEley, 8), KE)
        * u[gDof].reshape(nElex * nEley, 8)
    ).sum(1)
    obj.append((x**penal * ce).sum())
    dc = x ** (penal - 1) * ce

    # Sensitivity filtering
    dc = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0]

    # Stabilisation
    if loop > 1:
        dc = (dc + olddc) / 2

    # Add or delete element
    l1 = np.min(dc)
    l2 = np.max(dc)
    xnew = np.zeros(nElex * nEley)

    while (l2 - l1) / l2 > 1e-5:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(
            1e-5,
            np.sign(dc - lmid),
        )
        if np.sum(xnew) - vol * (nElex * nEley) > 0:
            l1 = lmid
        else:
            l2 = lmid

    # Skeletonize by scikit-image
    # Binary image data cleaning
    d = dMin * ((1 - vol) / (1 - volFrac)) ** 20

    imData = (xnew == 1).reshape((nEley, nElex))
    centreline = morphology.skeletonize(imData)
    skeletonLine = morphology.remove_small_objects(centreline, min_size=d).flatten()

    for index, isCentreline in enumerate(skeletonLine):
        if isCentreline:
            centreX, centreY = eleCentroid[index]

            cirdMin = (eleCentroid[:, 0] - centreX) ** 2 + (
                eleCentroid[:, 1] - centreY
            ) ** 2 <= (d / 2) ** 2

            xnew[cirdMin] = 1.0

    x = xnew

    # Compute the change
    if loop > 10:
        change = abs(
            np.sum(obj[loop - 10 : loop - 5]) - np.sum(obj[loop - 5 : loop])
        ) / np.sum(obj[loop - 5 : loop])

    # Plot to screen
    im.set_array(x.reshape((nEley, nElex)))
    # plt.pause(0.01)
    ax.draw_artist(im)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

    # Write iteration history to screen
    print(
        "it.: {0} , obj.: {1:.4f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, 0.5 * obj[loop - 1], sum(x) / (nElex * nEley), change
        )
    )

# Make sure the plot stays and that the shell remains
print("Optimisation done!")

plt.show()
