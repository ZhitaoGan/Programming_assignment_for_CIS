"""
Utility Functions Module

This module contains utility functions for data processing and file operations.
Used throughout the CISPA assignment for reading data files and performing calculations.
"""

import numpy as np
from pathlib import Path
from programs.frame_transform import FrameTransform
from programs import pivot_calibration

def read_cal_data(file_path):
    """
    Read calibration data from a text file.
    
    Args:
        file_path (str): Path to the calibration data file
        
    Returns:
        np.ndarray: Array containing the calibration data
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split(',')
            N_D, N_A, N_C = map(int, first_line[0:3])
            data = np.loadtxt(file, delimiter=',')
        if data.shape[0] == (N_D + N_A + N_C):
            d_points = data[0:N_D]
            a_points = data[N_D:N_D+N_A]
            c_points = data[N_D+N_A:]
            return {"d_points": d_points, "a_points": a_points, "c_points": c_points}
        else:
            N_frames = int(first_line[3])
            frame_size = N_D + N_A + N_C
            frames = []
            for k in range(N_frames):
                start = k * frame_size
                end = (k+1) * frame_size
                block = data[start:end]
                d_points = block[0:N_D]
                a_points = block[N_D:N_D+N_A]
                c_points = block[N_D+N_A:]
                frames.append({"d_points": d_points, "a_points": a_points, "c_points": c_points})
            return {"frames": frames,"N_D": N_D,"N_A": N_A,"N_C": N_C}
    except Exception as e:
        print(f"Error reading calibration data from {file_path}: {e}")
        return None


def read_pivot_data(file_path):
    """
    Read pivot calibration data from a text file.
    
    Args:
        file_path (str): Path to the pivot data file
        
    Returns:
        np.ndarray: Array containing the pivot data
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split(',')
            N_G, N_frames = map(int, first_line[:2])
            data = np.loadtxt(file, delimiter=',')
        frames = [data[i*N_G:(i+1)*N_G] for i in range(N_frames)]
        return {"frames": frames,"N_G": N_G}
    except Exception as e:
        print(f"Error reading pivot data from {file_path}: {e}")
        return None


def read_optpivot(file_path):
    """
    Read optical pivot data from a text file.
    
    Args:
        file_path (str): Path to the optical pivot data file
        
    Returns:
        np.ndarray: Array containing the optical pivot data
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split(',')
            N_D, N_H, N_frames = map(int, first_line[:3])
            data = np.loadtxt(file, delimiter=',')
        frame_size = N_D + N_H
        frames = []
        for k in range(N_frames):
            start = k * frame_size
            end = (k+1) * frame_size
            block = data[start:end]
            d_points = block[:N_D]
            h_points = block[N_D:]
            frames.append({"d_points": d_points, "h_points": h_points})
        return {"frames": frames,"N_D": N_D, "N_H": N_H}
    except Exception as e:
        print(f"Error reading optical pivot data from {file_path}: {e}")
        return None


def parse_files(output_file, cal_read, cal_body, output_dir="."):
    """
    Parse calibration files and generate output.
    
    Args:
        output_file (str): Name of the output file
        cal_read (np.ndarray): Calibration readings data
        cal_body (np.ndarray): Calibration body data
        output_dir (str): Output directory path
    """
    N_frames = len(cal_read["frames"])
    
    # Determine which registration to generate based on output filename
    if "Fa" in output_file:
        # Generate Fa registration (A to a points)
        F_list = []
        for frame in cal_read["frames"]:
            A = frame["a_points"]
            F_A = FrameTransform.Point_set_registration(A, cal_body["a_points"])
            # Extract transformation matrix (4x4 homogeneous matrix)
            T = np.eye(4)
            T[:3, :3] = F_A.rotation_matrix
            T[:3, 3] = F_A.translation_vector
            F_list.append(T.flatten())  # Flatten to 1D for saving
        np.savetxt(f"{output_dir}/{output_file}.txt", F_list, fmt="%.3f", delimiter=",")
    elif "Fd" in output_file:
        # Generate Fd registration (D to d points)
        F_list = []
        for frame in cal_read["frames"]:
            D = frame["d_points"]
            F_D = FrameTransform.Point_set_registration(D, cal_body["d_points"])
            # Extract transformation matrix (4x4 homogeneous matrix)
            T = np.eye(4)
            T[:3, :3] = F_D.rotation_matrix
            T[:3, 3] = F_D.translation_vector
            F_list.append(T.flatten())  # Flatten to 1D for saving
        np.savetxt(f"{output_dir}/{output_file}.txt", F_list, fmt="%.3f", delimiter=",")
    else:
        # Default: generate both registrations
        F_D_list, F_A_list = [], []
        for frame in cal_read["frames"]:
            D, A = frame["d_points"], frame["a_points"]
            F_D = FrameTransform.Point_set_registration(D, cal_body["d_points"])
            F_A = FrameTransform.Point_set_registration(A, cal_body["a_points"])
            
            # Extract transformation matrices
            T_D = np.eye(4)
            T_D[:3, :3] = F_D.rotation_matrix
            T_D[:3, 3] = F_D.translation_vector
            F_D_list.append(T_D.flatten())
            
            T_A = np.eye(4)
            T_A[:3, :3] = F_A.rotation_matrix
            T_A[:3, 3] = F_A.translation_vector
            F_A_list.append(T_A.flatten())
        
        # Save as separate .txt files
        np.savetxt(f"{output_dir}/Fd_a_registration.txt", F_D_list, fmt="%.2f", delimiter=",")
        np.savetxt(f"{output_dir}/Fa_a_registration.txt", F_A_list, fmt="%.2f", delimiter=",")


def C_expected(cal_body, output_file, input_Fa, input_Fd, output_dir="."):
    """
    Calculate expected C values and generate NAME-OUTPUT1.TXT format file.
    
    Args:
        cal_body (np.ndarray): Calibration body data
        output_file (str): Name of the output file
        input_Fa (str): Fa registration file
        input_Fd (str): Fd registration file
        output_dir (str): Output directory path
    """
    # Load Fa registration matrices (from .txt file)
    F_A_matrices = np.loadtxt(f"{output_dir}/{input_Fa}.txt", delimiter=",")
    N_frames = F_A_matrices.shape[0]
    
    # Load Fd registration matrices (from .txt file)  
    F_D_matrices = np.loadtxt(f"{output_dir}/{input_Fd}.txt", delimiter=",")
    
    # Reshape matrices back to 4x4
    F_A_list = [F_A_matrices[i].reshape(4, 4) for i in range(N_frames)]
    F_D_list = [F_D_matrices[i].reshape(4, 4) for i in range(N_frames)]
    
    # Get EM and optical pivot tip positions (from previous calibration results)
    # Extract dataset name from output_file (e.g., "pa1-debug-a-output1" -> "PA1-DEBUG-A")
    dataset_name = output_file.replace("-output1", "").upper().replace("-", "-")
    if dataset_name.startswith("PA1-DEBUG-"):
        dataset_prefix = dataset_name
    elif dataset_name.startswith("PA1-UNKNOWN-"):
        dataset_prefix = dataset_name
    else:
        # Fallback for other naming conventions
        dataset_prefix = f"PA1-DEBUG-{dataset_name[-1].upper()}"
    
    em_pivot_result = np.loadtxt(f"{output_dir}/{dataset_prefix}_EM_pivot.txt", delimiter=",")
    opt_pivot_result = np.loadtxt(f"{output_dir}/{dataset_prefix}_Optpivot.txt", delimiter=",")
    
    em_tip = em_pivot_result[1]  # Second row is pivot point (actual tip position)
    opt_tip = opt_pivot_result[1]  # Second row is pivot point (actual tip position)
    
    # Calculate expected C coordinates for each frame
    C_expected_all = []
    N_C = len(cal_body["c_points"])
    
    for k, (F_D, F_A) in enumerate(zip(F_D_list, F_A_list)):
        # Create FrameTransform objects from matrices
        F_D_transform = FrameTransform(F_D[:3, :3], F_D[:3, 3])
        F_A_transform = FrameTransform(F_A[:3, :3], F_A[:3, 3])
        
        # Calculate expected C coordinates: F_D * F_A^(-1) * c_points
        C_expected = F_D_transform.compose(F_A_transform.inverse()).transform_points(cal_body["c_points"])
        C_expected_all.append(C_expected)
    
    # Write output file in NAME-OUTPUT1.TXT format
    with open(f"{output_dir}/{output_file}.txt", 'w') as f:
        # Header line
        f.write(f"{N_C}, {N_frames}, {output_file}.txt\n")
        
        # EM pivot tip position
        f.write(f"{em_tip[0]:.3f}, {em_tip[1]:.3f}, {em_tip[2]:.3f}\n")
        
        # Optical pivot tip position  
        f.write(f"{opt_tip[0]:.3f}, {opt_tip[1]:.3f}, {opt_tip[2]:.3f}\n")
        
        # Expected C coordinates for each frame
        for frame_idx in range(N_frames):
            for marker_idx in range(N_C):
                c_coords = C_expected_all[frame_idx][marker_idx]
                f.write(f"{c_coords[0]:.3f}, {c_coords[1]:.3f}, {c_coords[2]:.3f}\n")


def em_pivot(empivot, output_file1, output_dir="."):
    """
    Perform EM pivot calibration and save results.
    
    Args:
        empivot (np.ndarray): EM pivot data
        output_file1 (str): Name of the output file
        output_dir (str): Output directory path
    """
    
    result = pivot_calibration.em_pivot_calibration(empivot)
    p_tip = result['tip_position']
    p_pivot = result['pivot_point']
    residual_error = result['residual_error']

    np.savetxt(f"{output_dir}/{output_file1}.txt", np.vstack([p_tip, p_pivot, [residual_error,0,0]]), fmt="%.3f", delimiter=",")


def opt_pivot(optpivot, cal_body, output_file2, output_dir="."):
    """
    Perform optical pivot calibration and save results.
    
    Args:
        optpivot (np.ndarray): Optical pivot data
        cal_body (np.ndarray): Calibration body data
        output_file2 (str): Name of the output file
        output_dir (str): Output directory path
    """
    # Use the new integrated optical pivot calibration with EM geometry
    result = pivot_calibration.opt_pivot_calibration(optpivot, cal_body)
    p_tip = result['tip_position']
    p_pivot = result['pivot_point']
    residual_error = result['residual_error']
    np.savetxt(f"{output_dir}/{output_file2}.txt", np.vstack([p_tip, p_pivot, [residual_error,0,0]]), fmt="%.3f", delimiter=",")


# Create a utility_functions object that contains all the functions
# This allows the import statement to work as expected
class UtilityFunctions:
    """Container class for utility functions."""
    
    def __init__(self):
        self.read_cal_data = read_cal_data
        self.read_pivot_data = read_pivot_data
        self.read_optpivot = read_optpivot
        self.parse_files = parse_files
        self.C_expected = C_expected
        self.em_pivot = em_pivot
        self.opt_pivot = opt_pivot


# Create an instance to be imported
utility_functions = UtilityFunctions()
