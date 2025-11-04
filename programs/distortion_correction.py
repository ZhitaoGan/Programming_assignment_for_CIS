"""
Distortion Correction Module

This module implements polynomial-based distortion correction for EM tracker measurements
using 3D Bernstein (Bezier) polynomials as described in CIS lectures.

The distortion correction maps distorted coordinates u = (x, y, z) to corrected coordinates
p̂ = (x̂, ŷ, ẑ) using a tensor-product Bernstein polynomial fit.
"""

import numpy as np
from scipy.special import comb


def bernstein_basis(t, i, n):
    """
    Compute single Bernstein basis polynomial B_i^(n)(t).

    B_i^(n)(t) = C(n,i) * t^i * (1-t)^(n-i)
    where C(n,i) is the binomial coefficient "n choose i".

    Args:
        t (float or np.ndarray): Parameter value(s) in [0, 1]
        i (int): Basis index (0 to n)
        n (int): Polynomial degree

    Returns:
        float or np.ndarray: Bernstein basis value(s)
    """
    return comb(n, i, exact=True) * (t ** i) * ((1 - t) ** (n - i))


def scale_to_box(points, bounds=None):
    """
    Scale coordinates to [0, 1]^3 box (ScaleToBox operation from slides).

    Args:
        points (np.ndarray): Input points, shape (N, 3)
        bounds (dict, optional): Pre-computed bounds {'min': array, 'max': array}

    Returns:
        tuple: (scaled_points, bounds_dict)
    """
    if bounds is None:
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bounds = {'min': min_coords, 'max': max_coords}
    else:
        min_coords = bounds['min']
        max_coords = bounds['max']

    # Avoid division by zero
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1.0

    # Scale to [0, 1]
    scaled = (points - min_coords) / range_coords

    return scaled, bounds


def scale_from_box(scaled_points, bounds):
    """
    Scale coordinates back from [0, 1]^3 to original range.

    Args:
        scaled_points (np.ndarray): Scaled points in [0, 1]
        bounds (dict): Dictionary with 'min' and 'max' arrays

    Returns:
        np.ndarray: Points in original scale
    """
    min_coords = bounds['min']
    max_coords = bounds['max']
    range_coords = max_coords - min_coords

    return scaled_points * range_coords + min_coords


def fit_distortion_correction(distorted_points, expected_points, degree=5):
    """
    Fit 3D Bernstein polynomial distortion correction using least squares.

    Fits the model:
        x̂(u) = Σ_{i=0}^n Σ_{j=0}^n Σ_{k=0}^n c_ijk^(x) B_i^(n)(u_x) B_j^(n)(u_y) B_k^(n)(u_z)

    and similarly for ŷ and ẑ, where u = (u_x, u_y, u_z) are scaled distorted coordinates.

    Args:
        distorted_points (np.ndarray): Measured (distorted) coordinates, shape (N, 3)
        expected_points (np.ndarray): True (expected) coordinates, shape (N, 3)
        degree (int): Polynomial degree n (default: 5)

    Returns:
        dict: Model with 'coefficients' (list of 3 arrays), 'degree', 'bounds'
    """
    N = len(distorted_points)
    n = degree
    n_coeffs = (n + 1) ** 3  # Total coefficients per coordinate

    # Scale distorted coordinates to [0, 1]^3
    u_scaled, bounds = scale_to_box(distorted_points)

    # Build design matrix A where each row is {B_i(u_x,s) B_j(u_y,s) B_k(u_z,s)}
    A = np.zeros((N, n_coeffs))

    for s in range(N):
        u_x, u_y, u_z = u_scaled[s]

        # Compute all Bernstein basis values for this point
        basis_x = np.array([bernstein_basis(u_x, i, n) for i in range(n + 1)])
        basis_y = np.array([bernstein_basis(u_y, j, n) for j in range(n + 1)])
        basis_z = np.array([bernstein_basis(u_z, k, n) for k in range(n + 1)])

        # Fill row with tensor products B_i * B_j * B_k
        col_idx = 0
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    A[s, col_idx] = basis_x[i] * basis_y[j] * basis_z[k]
                    col_idx += 1

    # Solve three separate least squares problems (one per output coordinate)
    coefficients = []
    for coord_idx in range(3):
        # Target values for this coordinate (x̂, ŷ, or ẑ)
        b = expected_points[:, coord_idx]

        # Solve A * c = b
        c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        coefficients.append(c)

    return {
        'coefficients': coefficients,  # [c^(x), c^(y), c^(z)]
        'degree': n,
        'bounds': bounds
    }


def apply_distortion_correction(distorted_points, correction_model):
    """
    Apply distortion correction to measured points.

    Evaluates the fitted Bernstein polynomial:
        p̂ = Σ_{ijk} c_ijk B_i(u_x) B_j(u_y) B_k(u_z)

    Args:
        distorted_points (np.ndarray): Distorted measurements, shape (N, 3) or (3,)
        correction_model (dict): Model from fit_distortion_correction()

    Returns:
        np.ndarray: Corrected coordinates, same shape as input
    """
    # Handle single point input
    single_point = False
    if distorted_points.ndim == 1:
        distorted_points = distorted_points.reshape(1, -1)
        single_point = True

    N = len(distorted_points)
    n = correction_model['degree']
    coefficients = correction_model['coefficients']
    bounds = correction_model['bounds']

    # Scale to [0, 1]^3 box
    u_scaled, _ = scale_to_box(distorted_points, bounds)

    # Evaluate polynomial for each point
    corrected = np.zeros((N, 3))

    for s in range(N):
        u_x, u_y, u_z = u_scaled[s]

        # Compute Bernstein basis values
        basis_x = np.array([bernstein_basis(u_x, i, n) for i in range(n + 1)])
        basis_y = np.array([bernstein_basis(u_y, j, n) for j in range(n + 1)])
        basis_z = np.array([bernstein_basis(u_z, k, n) for k in range(n + 1)])

        # Evaluate triple sum for each output coordinate
        for coord_idx in range(3):
            c = coefficients[coord_idx]
            value = 0.0
            col_idx = 0

            for i in range(n + 1):
                for j in range(n + 1):
                    for k in range(n + 1):
                        value += c[col_idx] * basis_x[i] * basis_y[j] * basis_z[k]
                        col_idx += 1

            corrected[s, coord_idx] = value

    # Return in original format
    if single_point:
        return corrected[0]

    return corrected


def correct_frame_markers(frame_data, correction_model):
    """
    Apply distortion correction to all markers in a frame.

    Args:
        frame_data (np.ndarray): Frame markers, shape (N_markers, 3)
        correction_model (dict): Distortion correction model

    Returns:
        np.ndarray: Corrected frame markers, shape (N_markers, 3)
    """
    return apply_distortion_correction(frame_data, correction_model)
