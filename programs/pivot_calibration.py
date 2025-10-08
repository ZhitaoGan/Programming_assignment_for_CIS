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

    # residual error
    residual_error = np.linalg.norm(R_stack @ x - p_stack)
    return p_tip, p_pivot, residual_error

def pivot_calibration(pivot_data):
    """
    Perform pivot calibration using the least squares method
    
    Args:
        pivot_data (dict): Dictionary containing pivot calibration data
            - 'tip_positions': List of tool tip positions (Nx3 array)
            - 'orient_frames': List of tool orientations (Nx3x3 array)
    
    Returns:
        dict: Dictionary containing calibration results
            - 'tip_position': 3x1 array representing the tip position
            - 'pivot_point': 3x1 array representing the pivot point
            - 'residual_error': Scalar representing the residual error
    """
    # TODO: Implement pivot calibration algorithm
    # This typically involves:
    # 1. Setting up the system of equations: R_i * p + t_i = d_i
    # 2. Solving for the pivot point p using least squares
    # 3. Computing the residual error
    
    tip_positions = pivot_data.get('tip_positions', [])
    orient_frames = pivot_data.get('orient_frames', [])
    
    if not tip_positions or not orient_frames:
        raise ValueError("Pivot data must contain both tip positions and orient frames")
    
    # solve for the tip position and pivot point
    p_list = [np.asarray(p, dtype=float) for p in tip_positions]
    R_list = [np.asarray(R, dtype=float) for R in orient_frames]
    tip_position, pivot_point, residual_error = solve_for_pivot(R_list, p_list)

    return {
        'tip_position': tip_position,
        'pivot_point': pivot_point,
        'residual_error': residual_error
    }


def em_pivot_calibration(em_pivot_data):
    """
    Perform EM (Electromagnetic) pivot calibration.
    
    Args:
        em_pivot_data (np.ndarray): EM pivot calibration data
        
    Returns:
        dict: Calibration results
    """
    # TODO: Implement EM pivot calibration
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


def opt_pivot_calibration(opt_pivot_data, cal_body_data=None):
    """
    Optical pivot calibration with integrated optical→EM transformation.
    
    This implementation follows the three-step process:
    1. Use EM-base geometry from CALBODY to estimate optical→EM transformation
    2. Transform probe points into EM coordinates per frame
    3. Solve pivot least squares in EM coordinate system
    
    Args:
        opt_pivot_data: Dictionary with "frames" containing optical measurements
        cal_body_data: Dictionary with "d_points" containing EM-base geometry
    Returns:
        tip_position (3,), pivot_point (3,), residual_error (float)
    """
    import numpy as np
    from programs.frame_transform import FrameTransform

    frames = opt_pivot_data["frames"]
    assert len(frames) >= 2
    
    # Step 1: Get EM-base geometry from CALBODY
    if cal_body_data is None:
        # Fallback: use first frame as EM geometry (original approach)
        d_em = np.asarray(frames[0]["d_points"], float)
    else:
        # Use actual EM-base geometry from calibration body
        d_em = np.asarray(cal_body_data["d_points"], float)
    
    # Get probe local shape from first frame (zero-mean)
    H0 = np.asarray(frames[0]["h_points"], float)
    h_local = H0 - H0.mean(axis=0)   # Probe local marker shape

    F_H_in_EM_list = []

    # Step 2: For each optical pivot frame
    for fr in frames:
        D_optical = np.asarray(fr["d_points"], float)  # Optical measurements of EM-base markers
        H_optical = np.asarray(fr["h_points"], float)  # Optical measurements of probe markers
        
        # Correspondence-based estimation: optical → EM
        # Use correspondences between optical D (measured) and EM D geometry (d_em)
        F_0_to_EM = FrameTransform.Point_set_registration(D_optical, d_em)
        
        # Transform probe H optical points into EM coordinates
        H_em = F_0_to_EM.transform_points(H_optical)
        
        # Register probe local shape to H_em to get F_H_in_EM[k]
        F_H_in_EM = FrameTransform.Point_set_registration(h_local, H_em)
        F_H_in_EM_list.append(F_H_in_EM)

    # Step 3: Stack and solve the pivot least squares
    R_list = [T.rotation_matrix for T in F_H_in_EM_list]
    p_list = [T.translation_vector for T in F_H_in_EM_list]

    K = len(R_list)
    A = np.zeros((3*K, 6))
    b = np.zeros((3*K,))
    I = np.eye(3)

    for k, (Rk, pk) in enumerate(zip(R_list, p_list)):
        A[3*k:3*k+3, :3] = Rk
        A[3*k:3*k+3, 3:] = -I
        b[3*k:3*k+3] = -pk

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    t_tip = x[:3]         # probe tip in probe frame
    p_pivot_em = x[3:]    # pivot in EM-base frame

    # Residual RMS
    res = (A @ x - b).reshape(-1, 3)
    rms = float(np.sqrt((res**2).sum(axis=1).mean()))

    return {
        "tip_position": t_tip,
        "pivot_point": p_pivot_em,
        "residual_error": rms,
    }
