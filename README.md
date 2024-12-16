# From Topology Optimisation to Discrete Members: Feature Extraction and Reconstruction

## 🐍 Python code

The Python code is comprised of four principal components: topology optimisation (BESO method) (`topOptBESO.py`), feature extraction based on principal stress trajectories (`finiteEleAnalysis.py`), truss structure reconstruction (`trussTop.py`), and validation (`structuralAnalysis.py`). Communication between the various components is facilitated by the utilisation of text files.

### ✂️ Topology optimisation

The topology optimisation is conducted using the BESO method, and the code is adapted from the MATLAB code developed by Huang & Xie at RMIT and implemented in Python.

### 🎗️ Feature extraction

The process of feature extraction is based on the tracing of principal stress trajectories. The code incorporates both stress analysis and displacement analysis. Two visualisations of the stress tensor are employed: principal stress tensor glyphs and principal stress trajectories.

### 🏗️ Truss reconstruction

The truss topology is constructed through the implementation of a breadth-first search (BSF) algorithm.

### 🔬 Validation

A finite element method appropriate for two-dimensional truss structure is employed to validate the final results.

## 🔢 MATLAB code

The heuristic method employed for result validation have been implemented in MATLAB. The MATLAB code comprises seven files.
