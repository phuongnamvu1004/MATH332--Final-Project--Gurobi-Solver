# math332-final-prj--gurobi-nonlinear-opt-solver

A Python API nonlinear optimization solver using Gurobi.

## Running the Code

- Setting up the environment:
```bash
conda create -n gurobi-nonlinear-opt-solver python=3.11 -y
conda activate gurobi-nonlinear-opt-solver
cd src/
pip install -r requirements.txt
```
(for Gurobi, obtain the academic license if needed)
- Running the solver:
```bash
python3 compute_stats.py # set the computed stats like expected returns and covariance matrix
python3 gurobi_solver.py # run the solver
python3 duality_solver.py # run the duality solver and verify p* = d*
```
