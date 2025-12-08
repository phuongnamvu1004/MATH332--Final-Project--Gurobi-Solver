"""Utility functions to load historical return data and compute
expected returns and covariance matrix for the Markowitz model.

We use the discounted log-mean estimator from the project write-up:

    r_j = exp( sum_{t=1}^T p^{T-t} log R_j(t) / sum_{t=1}^T p^{T-t} ),

where p in (0, 1] is a discount factor that emphasizes more recent
observations when p < 1 and reduces to the simple log-mean when p = 1.

The covariance matrix is computed from (centered) log-returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ReturnData:
    """Container for preprocessed return data.

    Attributes
    ----------
    years : np.ndarray
        1D array of shape (T,) containing the time index (e.g., years).
    asset_names : List[str]
        Names of the assets, taken from the non-year columns of the CSV.
    returns : np.ndarray
        2D array of gross returns with shape (T, n), where T is the
        number of periods and n is the number of assets.
    """

    years: np.ndarray
    asset_names: List[str]
    returns: np.ndarray


def load_returns(csv_path: str, year_col: str = "Year") -> ReturnData:
    """Load historical gross returns from a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing returns.
    year_col : str, default "Year"
        Name of the column containing the time index (ignored when
        computing statistics). If not present, all columns are treated
        as asset returns and the years array will be a simple index
        0, 1, ..., T-1.

    Returns
    -------
    ReturnData
        A dataclass with years, asset_names, and returns matrix.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if year_col in df.columns:
        years = df[year_col].to_numpy()
        asset_cols = [c for c in df.columns if c != year_col]
    else:
        years = np.arange(len(df))
        asset_cols = list(df.columns)

    if not asset_cols:
        raise ValueError("No asset columns found in returns CSV.")

    returns = df[asset_cols].to_numpy(dtype=float)

    return ReturnData(years=years, asset_names=asset_cols, returns=returns)


def compute_discounted_log_mean(returns: np.ndarray, p: float) -> np.ndarray:
    """Compute discounted log-mean expected returns for each asset.

    Parameters
    ----------
    returns : np.ndarray
        2D array of gross returns with shape (T, n).
    p : float
        Discount factor in (0, 1]. Values closer to 1 give nearly equal
        weight to all periods; smaller values emphasize recent periods.

    Returns
    -------
    np.ndarray
        1D array of shape (n,) containing the estimated expected
        gross return r_j for each asset.
    """

    if not (0.0 < p <= 1.0):
        raise ValueError("Discount factor p must lie in (0, 1].")

    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n).")

    T, n = returns.shape

    # Prevent log of non-positive values.
    if np.any(returns <= 0):
        raise ValueError("All gross returns must be strictly positive to take logs.")

    # weights for t = 0, ..., T-1 corresponding to p^{T-1-t}
    exponents = np.arange(T - 1, -1, -1, dtype=float)
    weights = p ** exponents
    weights /= weights.sum()

    log_returns = np.log(returns)  # shape (T, n)

    # Weighted average of log-returns along the time axis.
    weighted_log_means = (weights[:, None] * log_returns).sum(axis=0)

    # Map back via exp to obtain gross expected returns.
    r = np.exp(weighted_log_means)
    return r


def compute_covariance_log_returns(returns: np.ndarray) -> np.ndarray:
    """Compute covariance matrix of log-returns.

    Parameters
    ----------
    returns : np.ndarray
        2D array of gross returns with shape (T, n).

    Returns
    -------
    np.ndarray
        2D covariance matrix C of shape (n, n).
    """

    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, n).")

    if np.any(returns <= 0):
        raise ValueError("All gross returns must be strictly positive to take logs.")

    log_returns = np.log(returns)

    # np.cov expects variables in columns when rowvar=False.
    C = np.cov(log_returns, rowvar=False, bias=False)
    return C


def compute_stats(csv_path: str, p: float = 1.0, year_col: str = "Year") -> Tuple[List[str], np.ndarray, np.ndarray]:
    """High-level helper to go from CSV to (asset_names, r, C).

    Parameters
    ----------
    csv_path : str or Path
        Path to the returns CSV.
    p : float, default 1.0
        Discount factor for expected return estimation.
    year_col : str, default "Year"
        Name of the year/time index column.

    Returns
    -------
    asset_names : list of str
        The names of the assets.
    r : np.ndarray
        1D array of shape (n,) with discounted log-mean expected returns.
    C : np.ndarray
        2D covariance matrix of log-returns with shape (n, n).
    """

    data = load_returns(csv_path, year_col=year_col)
    r = compute_discounted_log_mean(data.returns, p=p)
    C = compute_covariance_log_returns(data.returns)
    return data.asset_names, r, C


if __name__ == "__main__":
    default_csv = "../data/returns.csv"

    if Path(default_csv).exists():
        assets, r, C = compute_stats(default_csv, p=0.9) # Example with p=0.9, no p defined will mean p=1.0 (equal weighting)
        print("Loaded assets:", assets)
        print("Expected returns r shape:", r.shape)
        print("Expected returns r:", r)
        print("Covariance matrix C shape:", C.shape)
        print("Covariance matrix C:", C)
    else:
        print(f"No default CSV found at {default_csv}. Please provide a valid path.")
