"""
rotation_utils.py — Completed Rotation Utilities (from Module 0)
=================================================================

This file contains completed implementations of the random rotation functions
you built in Module 0. They are provided here as a ready-to-import dependency
for Module 2's TurboQuant_mse pipeline.

The core function is `random_rotation_matrix(d, seed)`, which generates a
Haar-distributed (uniformly random) d×d orthogonal matrix Π using QR
decomposition of a random Gaussian matrix.

Mathematical guarantee (proved in Module 0):
    For any unit vector x ∈ R^d:  Π·x is uniform on S^{d-1}
    Each coordinate (Π·x)_i ~ Beta((d-1)/2, (d-1)/2) on [−1, 1]
    As d → ∞:                     Beta → N(0, 1/d)

Reference: Module 0, Exercise 02 — "How Random Rotation Creates a Universal
Distribution"
"""

import numpy as np


def random_rotation_matrix(d: int, seed: int = None) -> np.ndarray:
    """
    Generate a uniformly random d×d rotation (orthogonal) matrix Π.

    A rotation matrix satisfies Π^T Π = I and det(Π) = +1.
    Multiplying any unit vector x by Π places Π·x uniformly on S^{d-1}.

    Algorithm (Haar measure via QR decomposition):
        1. Sample G: a d×d matrix with i.i.d. N(0,1) entries.
        2. Compute QR decomposition:  G = Q · R
           Q is orthogonal, R is upper-triangular.
        3. Adjust signs:  Q ← Q * sign(diag(R))
           This ensures the distribution is exactly Haar (uniform over O(d))
           and det(Q) = +1.

    Parameters
    ----------
    d    : int, dimension of the rotation matrix (will be d×d)
    seed : int or None, random seed for reproducibility

    Returns
    -------
    Pi : np.ndarray, shape (d, d), orthogonal matrix with det ≈ +1
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # Fix sign to ensure Haar distribution (det = +1)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1      # edge case: exactly 0 diagonal entry
    Pi = Q * signs[np.newaxis, :]
    return Pi


def rotate(Pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply rotation Π to a vector or batch of vectors.

    Single vector  x  of shape (d,):   returns Π @ x, shape (d,)
    Batch          X  of shape (N, d): returns X @ Π^T, shape (N, d)

    Parameters
    ----------
    Pi : np.ndarray, shape (d, d), orthogonal rotation matrix
    x  : np.ndarray, shape (d,) or (N, d)

    Returns
    -------
    x_rot : np.ndarray, same shape as x, ||x_rot|| == ||x||
    """
    if x.ndim == 1:
        return Pi @ x
    else:
        return x @ Pi.T


def inverse_rotate(Pi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Apply inverse rotation Π^T (= Π^{-1} for orthogonal matrices).

    Parameters
    ----------
    Pi : np.ndarray, shape (d, d)
    y  : np.ndarray, shape (d,) or (N, d)

    Returns
    -------
    y_unrot : np.ndarray, same shape as y
    """
    if y.ndim == 1:
        return Pi.T @ y
    else:
        return y @ Pi
