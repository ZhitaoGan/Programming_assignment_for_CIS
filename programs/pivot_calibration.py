"""
Pivot Calibration Module

This module contains functions for pivot calibration calculations.
Used for determining the pivot point of a tracked tool.
"""

import numpy as np
from programs.frame_transform import FrameTransform

# helper function to calculate the pivot point
def as_frames(data):
    """
    return a list of (N,3) arrays, where each array is a frame of data
    
    Args:
        data (dict): Dictionary containing pivot calibration data
    """
    if isinstance(data, dict) and "frames" in data:
        return [np.asarray(f, dtype=float) for f in data["frames"]]
    return [np.asarray(f, dtype=float) for f in data]
def first_frame_centroid(frames):
    """
    Used the first frame to define the local coordinate system
    return the local coordinate system
    
    Args:
        frames (list): List of (N,3) arrays
    """
    first_frame = np.asarray(frames[0], dtype=float)
    centroid = np.mean(first_frame, axis=0)
    g_local = first_frame - centroid
    return g_local

def poses_from_frames(local_coord, frames):
    """
    return lists of rotation matrices and translation vectors
    
    Args:
        local_coord (np.ndarray): Local coordinate system
        frames (list): List of (N,3) arrays
    """
    R_list, p_list = [], []
    for frame in frames:
        pose = FrameTransform.Point_set_registration(local_coord, frame)
        R_list.append(pose.rotation_matrix)
        p_list.append(pose.translation_vector)
    return R_list, p_list

def solve_for_pivot(R_list, p_list):
    """
    return the tip position and pivot point
    
    Args:
        R_list (list): List of rotation matrices
        p_list (list): List of translation vectors
    """
    # Stack rotation matrices and identity matrices
    R_stack = np.vstack([np.hstack([R, -np.eye(3)]) for R in R_list])
    p_stack = -np.concatenate(p_list)
    
    # Solve least squares problem
    x, residuals, rank, s = np.linalg.lstsq(R_stack, p_stack, rcond=None)
    p_tip = x[:3]
    p_pivot = x[3:]

    # Calculate RMS residual error
    residual_vector = R_stack @ x - p_stack
    # Reshape to (num_frames, 3) and compute RMS per frame
    residual_per_frame = residual_vector.reshape(-1, 3)
    residual_error = np.sqrt(np.mean(np.sum(residual_per_frame**2, axis=1)))
    return p_tip, p_pivot, residual_error

def em_pivot_calibration(em_pivot_data):
    """
    Perform EM (Electromagnetic) pivot calibration.
    
    Args:
        em_pivot_data (np.ndarray): EM pivot calibration data
        
    Returns:
        dict: Calibration results
    """
    frames = as_frames(em_pivot_data)
    if len(frames) < 2:
        raise ValueError("EM pivot data must contain at least 2 frames")
    local_coord = first_frame_centroid(frames)
    R_list, p_list = poses_from_frames(local_coord, frames)
    tip_position, pivot_point, residual_error = solve_for_pivot(R_list, p_list)
    return {
        'tip_position': tip_position,
        'pivot_point': pivot_point,
        'residual_error': residual_error
    }


def opt_pivot_calibration(opt_pivot_data, cal_body_data):
    """
    Optical pivot calibration with integrated optical→EM transformation.

    This implementation follows the three-step process:
    1. Use EM-base geometry from CALBODY to estimate optical→EM transformation
    2. Transform probe points into EM coordinates per frame
    3. Solve pivot least squares in EM coordinate system using helper functions

    Args:
        opt_pivot_data: Dictionary with "frames" containing optical measurements
        cal_body_data: Dictionary with "d_points" containing EM-base geometry (required)
    Returns:
        dict: Calibration results with tip_position, pivot_point, residual_error
    """
    frames = opt_pivot_data["frames"]
    assert len(frames) >= 2

    # Step 1: Get EM-base geometry from CALBODY (required for optical→EM transformation)
    d_em = np.asarray(cal_body_data["d_points"], float)

    # Step 2: Define local probe coordinate system using helper function
    # Extract h_points from frames to create probe frames
    h_frames = [np.asarray(fr["h_points"], float) for fr in frames]
    h_local = first_frame_centroid(h_frames)

    # Step 3: Transform optical measurements to EM coordinates and compute poses
    h_em_frames = []
    for fr in frames:
        D_optical = np.asarray(fr["d_points"], float)  # Optical measurements of EM-base markers
        H_optical = np.asarray(fr["h_points"], float)  # Optical measurements of probe markers

        # Correspondence-based estimation: optical → EM
        # Use correspondences between optical D (measured) and EM D geometry (d_em)
        F_0_to_EM = FrameTransform.Point_set_registration(D_optical, d_em)

        # Transform probe H optical points into EM coordinates
        H_em = F_0_to_EM.transform_points(H_optical)
        h_em_frames.append(H_em)

    # Step 4: Use helper functions to compute poses and solve pivot calibration
    R_list, p_list = poses_from_frames(h_local, h_em_frames)
    tip_position, pivot_point, residual_error = solve_for_pivot(R_list, p_list)

    return {
        "tip_position": tip_position,
        "pivot_point": pivot_point,
        "residual_error": residual_error,
    }
