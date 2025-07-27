# Fault-tolerant interfaces for modular quantum computing on diverse qubit platforms

This folder contains code used for validation and illustration for the research article *Fault-tolerant interfaces for modular quantum computing on diverse qubit platforms* (Manuscript in preparation). This project directory is self-contained.

## Contents

- **`ConstantRateDistillation/`**: Code for exactly evaluating the performance of classical codes from [gefenbaranes](https://github.com/gefenbaranes/ConstantRateDistillation).
- **`data/`**: Contains all optimal sequences found by DFS used to generate the figures for the article, including static data for physical distillation.
- **`rate_plot.ipynb`**: Generates plots and figures related to the article.
- **`compute_rate_data.py`**: Functions used by `rate_plot.ipynb` to compute figure data.
- **`interactive_plot.py`**: Makes interactive figure for exploring parameter space based on precomputed figure data.
- **`sequence_class.py`**: The main class object used to represent and analyse logical distillation sequences.
- **`sequence_optimisation.py`**: Optimised DFS for finding optimal sequences.
- **`parallel_optim_search.py`**: Script for running reduced DFS as described in the appendices.
- **`parallel_full_search.py`**: Script for running multiple DFSs in parallel.
- **`sequence_simulation.py`**: Monte-carlo simulation of balanced sequence pipeline.
- **`physical_distillation.py`**: Code for studying physical distillation (needs clean-up).
- **`utils.py`**: Helper functions.

## Python Version

This project was developed and tested with **Python 3.12.3**. 