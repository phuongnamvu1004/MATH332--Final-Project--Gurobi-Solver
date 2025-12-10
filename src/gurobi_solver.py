from __future__ import annotations

from typing import List, Dict, Any, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_markowitz_model(
    assets: List[str],
    r: np.ndarray,
    C: np.ndarray,
    mu: float,
) -> Tuple[gp.Model, Dict[str, gp.Var]]:
    """
    Build the Markowitz portfolio optimization model:

        min_x  - r^T x + mu x^T C x
        s.t.   sum_j x_j = 1
               x_j >= 0

    Parameters
    ----------
    assets : list of asset names, length n
    r      : expected returns vector, shape (n,)
    C      : covariance matrix, shape (n, n)
    mu     : risk aversion parameter (>= 0)

    Returns
    -------
    model  : Gurobi model
    x_vars : dict mapping asset name -> Gurobi variable x_j
    """
    n = len(assets)
    assert r.shape[0] == n, "Length of r must match number of assets"
    assert C.shape == (n, n), "C must be an (n, n) covariance matrix"

    model = gp.Model("markowitz_portfolio")

    # Decision variables: x_j for each asset j, fraction of wealth in asset j
    x_vars: Dict[str, gp.Var] = {
        assets[j]: model.addVar(lb=0.0, ub=1.0, name=f"x_{assets[j]}")
        for j in range(n)
    }

    model.update()

    # Full-investment constraint: sum_j x_j = 1
    model.addConstr(
        gp.quicksum(x_vars[name] for name in assets) == 1.0,
        name="full_investment",
    )

    # Expected return: r^T x
    expected_return_expr = gp.quicksum(
        r[j] * x_vars[assets[j]] for j in range(n)
    )

    # Risk: x^T C x
    risk_expr = gp.quicksum(
        C[i, j] * x_vars[assets[i]] * x_vars[assets[j]]
        for i in range(n)
        for j in range(n)
    )

    # Objective: min  - r^T x + mu x^T C x
    model.setObjective(-expected_return_expr + mu * risk_expr, GRB.MINIMIZE)

    return model, x_vars


def optimize_portfolio(
    assets: List[str],
    r: np.ndarray,
    C: np.ndarray,
    mu: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build and solve the Markowitz model for given data and mu."""
    model, x_vars = build_markowitz_model(assets, r, C, mu)

    if not verbose:
        model.Params.OutputFlag = 0

    model.optimize()

    result: Dict[str, Any] = {"status": model.Status}

    if model.Status == GRB.OPTIMAL:
        x_star = np.array([x_vars[name].X for name in assets])
        expected_return_val = float(np.dot(r, x_star))
        variance_val = float(x_star @ C @ x_star)
        std_dev_val = np.sqrt(variance_val)

        result.update(
            {
                "weights": {assets[j]: float(x_star[j]) for j in range(len(assets))},
                "expected_return": expected_return_val,
                "variance": variance_val,
                "std_dev": std_dev_val,
                "objective_value": float(model.ObjVal),
            }
        )

    return result


def solve_for_grid_of_mu(
    assets: List[str],
    r: np.ndarray,
    C: np.ndarray,
    mu_values: List[float],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Solve the Markowitz problem for a list of mu values to trace out the efficient frontier."""
    results: List[Dict[str, Any]] = []
    for mu in mu_values:
        res = optimize_portfolio(assets, r, C, mu=mu, verbose=verbose)
        res["mu"] = mu
        results.append(res)
    return results


if __name__ == "__main__":
    from compute_stats import compute_stats
    assets, r, C = compute_stats("../data/returns.csv", p=0.9)
    mu_values = [0.0, 1.0, 2.0, 4.0, 8.0, 1024.0]
    frontier = solve_for_grid_of_mu(assets, r, C, mu_values, verbose=False)
    for res in frontier:
        print(f"mu = {res['mu']}, E[R] = {res['expected_return']:.4f}, Std = {res['std_dev']:.4f}")
        print("Weights:")
        for asset, weight in res["weights"].items():
            print(f"{asset}\t{weight:.4f}")
        print("-" * 40)
    
    pass