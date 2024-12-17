# From Topology Optimisation to Discrete Members: Feature Extraction and Reconstruction

The core functionality of the project has been implemented in Python, while the heuristic method utilised for result validation has been developed in MATLAB.

## 🐍 Python code

The Python code consists of four main components: topology optimisation using the BESO method (`topOptBESO.py`), feature extraction based on principal stress trajectories (`finiteEleAnalysis.py`), truss structure reconstruction (`trussTop.py`), and result validation (`structuralAnalysis.py`). The implementation is designed to be semi-automated, allowing for step-by-step verification of each process. Data transfer between components is facilitated through text files for clarity and traceability.

### ✂️ Topology Optimisation Using BESO Method

The topology optimisation is performed using the BESO method. The implementation is adapted from the MATLAB code developed by Huang and Xie [@huangEvolutionaryTopologyOptimization2010] at RMIT and translated into Python. The output file, `EOutput.txt`, records the Young’s modulus values for each finite element, which are used as input for the subsequent stages.

### 🎗️ Feature Extraction Based on Principal Stress Trajectories

Feature extraction is carried out by tracing principal stress trajectories derived from stress analysis. This component incorporates both stress and displacement analyses to identify critical load paths. Two visualisation techniques are employed to represent the stress tensor: principal stress tensor glyphs and principal stress trajectories. The nodal coordinates required for reconstructing the truss structure are saved in `nodeOutput.txt`.

### 🏗️ Truss Structure Reconstruction

The truss topology is reconstructed using a BFS algorithm. This step processes the extracted stress trajectory data into a discrete truss structure suitable for further analysis.

### 🔬 Validation

The final results are validated using the finite element method specifically formulated for two-dimensional truss structures. Structural performance, including displacements and stress distributions, is evaluated to ensure the accuracy and feasibility of the reconstructed truss system.

## 🔢 MATLAB Code

The MATLAB implementation comprises seven files that support the heuristic validation process.

### 💡 Heuristic Optimisation

The main heuristic optimisation process is implemented in `trussOptHeu.m`, while the objective function is defined in `objectiveFuncHeu.m`.

### 📽️ Visualisation and Supporting Computations

The iterative optimisation process is handled by `trussOptHeuFigure.m`, which generates graphical outputs for analysis. The supporting files `trussDisp.m`, `trussMass.m`, and `trussStress.m` are responsible for computing the displacement, mass, and stress of the truss structure, respectively.

### 🪵 Baseline Truss Analysis

The file `originalTruss.m` contains the baseline finite element analysis (FEA) code for a two-dimensional truss structure. This serves as a reference for comparison and validation purposes.
