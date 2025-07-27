# Simulators

This folder containts tools for simulation of discrete-variable (DV), continuous-variable (CV), and Gottesman–Kitaev–Preskill (GKP) quantum systems. Initially developed for the purpose of investigating computations using GKP qubits for the published article *Impact of Finite Squeezing on Near-Term Quantum Computations Using GKP Qubits* [arXiv:2507.15955](https://arxiv.org/abs/2507.15955).

## Contents

- **`cv_simulator`**: Simulation of CV quantum computations based on functional matrix product states as described in *Functional matrix product state simulation of continuous variable quantum circuits* [arXiv:2504.05860](https://arxiv.org/abs/2504.05860)
- **`dv_simulator`**: Very simple qubit state vector simulator based on NumPy. Also includes `numpy_quantum.py` with generic linear algebra convenience functions for manipulating NumPy state vectors.
- **`gkp_simulator`**: Simulation of measurement-based GKP computations. Uses the CV simulator as backend.

## Python Version

This project was developed and tested with **Python 3.12.3**. 