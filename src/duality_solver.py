from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from compute_stats import compute_stats
from gurobi_solver import optimize_portfolio


def _make_spd_inverse(C: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """
    Ensure C is invertible (SPD) by adding a small ridge if needed, then invert.
    This is a practical step: the theoretical dual uses C^{-1}; with finite samples,
    C can be singular or ill-conditioned.
    """
    n = C.shape[0]
    C_reg = C.copy()

    # Symmetrize defensively (numerical noise)
    C_reg = 0.5 * (C_reg + C_reg.T)

    # Try direct inverse; if it fails, add ridge and retry
    try:
        return np.linalg.inv(C_reg)
    except np.linalg.LinAlgError:
        C_reg = C_reg + ridge * np.eye(n)
        return np.linalg.inv(C_reg)


def solve_dual_markowitz(
    assets: List[str],
    r: np.ndarray,
    C: np.ndarray,
    mu: float,
    ridge: float = 1e-10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve the dual of:
        min_x  -r^T x + mu x^T C x
        s.t.   1^T x = 1, x >= 0

    Dual (for mu > 0, assuming C invertible or regularized):
        maximize g(nu, lambda) = -(1/(4mu)) q^T C^{-1} q - nu
        s.t. lambda >= 0
        where q = r - nu*1 + lambda

    We solve the equivalent convex minimization:
        minimize f(nu, lambda) = (1/(4mu)) q^T C^{-1} q + nu
        s.t. lambda >= 0

    Then d* = max g = - min f.
    """
    if mu <= 0:
        raise ValueError("Dual QP here requires mu > 0. (mu=0 is a different, purely linear case.)")

    n = len(assets)
    assert r.shape == (n,)
    assert C.shape == (n, n)

    M = _make_spd_inverse(C, ridge=ridge)  # M = C^{-1}

    model = gp.Model("markowitz_dual")

    if not verbose:
        model.Params.OutputFlag = 0

    # Dual variables:
    # nu is free (one equality constraint)
    nu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu")

    # lambda >= 0 (one per inequality -x <= 0)
    lam = {
        assets[j]: model.addVar(lb=0.0, ub=GRB.INFINITY, name=f"lambda_{assets[j]}")
        for j in range(n)
    }

    model.update()

    # Build q_i = r_i - nu + lambda_i (since 1-vector multiplies nu)
    # Note: r_i is constant, nu and lambda_i are vars -> q_i is linear expression.
    q = []
    for j, name in enumerate(assets):
        qj = r[j] - nu + lam[name]  # LinExpr
        q.append(qj)

    # Quadratic form: q^T M q = sum_{i,j} M[i,j] q_i q_j
    quad = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            if M[i, j] != 0.0:
                quad.add(M[i, j] * q[i] * q[j])

    # Minimize f = (1/(4mu)) q^T M q + nu
    obj = (1.0 / (4.0 * mu)) * quad + nu
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    result: Dict[str, Any] = {"status": model.Status}

    if model.Status == GRB.OPTIMAL:
        nu_star = float(nu.X)
        lam_star = np.array([lam[name].X for name in assets], dtype=float)

        f_star = float(model.ObjVal)       # min f
        d_star = -f_star                   # max g

        result.update(
            {
                "nu": nu_star,
                "lambda": {assets[j]: float(lam_star[j]) for j in range(n)},
                "f_star_min": f_star,
                "dual_value_d_star": d_star,
            }
        )

    return result


def print_duality_gaps(csv_path: str, p: float, ridge: float = 1e-8, tol: float = 1e-7) -> None:
    assets, r, C = compute_stats(csv_path, p=p)
    mus = [0.0, 1.0, 2.0, 4.0, 8.0, 1024.0]

    for mu in mus:
        primal = optimize_portfolio(assets, r, C, mu=mu, verbose=False)
        if primal["status"] != GRB.OPTIMAL:
            print(f"mu={mu}: primal status {primal['status']}")
            continue
        p_star = float(primal["objective_value"])

        if mu == 0.0:
            d_star = -float(np.max(r))
        else:
            dual = solve_dual_markowitz(assets, r, C, mu=mu, ridge=ridge, verbose=False)
            if dual["status"] != GRB.OPTIMAL:
                print(f"mu={mu}: dual status {dual['status']}")
                continue
            d_star = float(dual["dual_value_d_star"])

        gap = p_star - d_star
        if abs(gap) < tol:
            gap = 0.0

        gap_str = "0" if gap == 0.0 else f"{gap:.3e}"
        print(f"mu = {mu}, p* = {p_star:.10f}, d* = {d_star:.10f}, p* - d* = {gap_str}")


def main():
    csv = "../data/returns.csv"
    p = 0.9
    print_duality_gaps(csv_path=csv, p=p, ridge=1e-10)


if __name__ == "__main__":
    main()