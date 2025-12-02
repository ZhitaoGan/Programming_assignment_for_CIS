"""
Utility Functions Module

This module contains utility functions for data processing and file operations.
Used throughout the CISPA assignment for reading data files and performing calculations.
"""

import numpy as np
from pathlib import Path
from programs.frame_transform import FrameTransform
from programs import pivot_calibration
from programs import distortion_correction

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


def parse_files(output_file, cal_read, cal_body, output_dir="output"):
    """
    Parse calibration files and generate output.

    Args:
        output_file (str): Name of the output file (must contain "Fa" or "Fd")
        cal_read (np.ndarray): Calibration readings data
        cal_body (np.ndarray): Calibration body data
        output_dir (str): Output directory path

    Raises:
        ValueError: If output_file doesn't contain "Fa" or "Fd"
    """
    # Validate output_file format
    if "Fa" not in output_file and "Fd" not in output_file:
        raise ValueError(f"output_file must contain 'Fa' or 'Fd', got: {output_file}")

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


def C_expected(cal_body, output_file, input_Fa, input_Fd, output_dir="output"):
    """
    Calculate expected C values and generate NAME-OUTPUT1.TXT format file.

    Args:
        cal_body (np.ndarray): Calibration body data
        output_file (str): Name of the output file (e.g., "pa1-debug-a-output1")
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

    # Extract dataset name from output_file (e.g., "pa1-debug-a-output1" -> "pa1-debug-a")
    dataset_prefix = output_file.replace("-output1", "")

    # Load pivot calibration results
    em_pivot_result = np.loadtxt(f"{output_dir}/{dataset_prefix}_EM_pivot.txt", delimiter=",")
    opt_pivot_result = np.loadtxt(f"{output_dir}/{dataset_prefix}_Optpivot.txt", delimiter=",")

    # Row 1 is pivot point (in tracker/EM base coordinates)
    em_pivot = em_pivot_result[1]
    opt_pivot = opt_pivot_result[1]
    
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

        # EM pivot point
        f.write(f"{em_pivot[0]:8.2f}, {em_pivot[1]:8.2f}, {em_pivot[2]:8.2f}\n")

        # Optical pivot point
        f.write(f"{opt_pivot[0]:8.2f}, {opt_pivot[1]:8.2f}, {opt_pivot[2]:8.2f}\n")

        # Expected C coordinates for each frame
        for frame_idx in range(N_frames):
            for marker_idx in range(N_C):
                c_coords = C_expected_all[frame_idx][marker_idx]
                f.write(f"{c_coords[0]:8.2f}, {c_coords[1]:8.2f}, {c_coords[2]:8.2f}\n")


def em_pivot(empivot, output_file1, output_dir="output"):
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


def opt_pivot(optpivot, cal_body, output_file2, output_dir="output"):
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


def read_output1(file_path):
    """
    Read output1 file and extract expected C marker positions.

    Args:
        file_path (str): Path to the output1 file

    Returns:
        dict: Dictionary containing:
            - 'N_C': Number of C markers
            - 'N_frames': Number of frames
            - 'em_pivot': EM pivot point coordinates
            - 'opt_pivot': Optical pivot point coordinates
            - 'c_expected_frames': List of arrays with expected C positions per frame
    """
    try:
        with open(file_path, 'r') as file:
            # Read header line
            first_line = file.readline().strip().split(',')
            N_C = int(first_line[0])
            N_frames = int(first_line[1])

            # Read pivot points
            em_pivot = np.array([float(x) for x in file.readline().strip().split(',')])
            opt_pivot = np.array([float(x) for x in file.readline().strip().split(',')])

            # Read all C coordinates
            data = np.loadtxt(file, delimiter=',')

        # Reshape into frames
        c_expected_frames = []
        for k in range(N_frames):
            start_idx = k * N_C
            end_idx = (k + 1) * N_C
            c_expected_frames.append(data[start_idx:end_idx])

        return {
            'N_C': N_C,
            'N_frames': N_frames,
            'em_pivot': em_pivot,
            'opt_pivot': opt_pivot,
            'c_expected_frames': c_expected_frames
        }
    except Exception as e:
        print(f"Error reading output1 file from {file_path}: {e}")
        return None


def extract_distortion_calibration_data(cal_readings, output1_data):
    """
    Extract paired data for distortion calibration.

    Extracts (distorted, expected) pairs: {C_i[k] -> C_i^(expected)[k]}
    where C_i[k] are the measured (distorted) C markers and C_i^(expected)[k]
    are the expected (true) positions from output1.

    Args:
        cal_readings (dict): Calibration readings data with 'frames'
        output1_data (dict): Output1 data from read_output1()

    Returns:
        tuple: (distorted_points, expected_points)
            - distorted_points: Array of measured C positions, shape (N_total, 3)
            - expected_points: Array of expected C positions, shape (N_total, 3)
    """
    N_frames = output1_data['N_frames']

    distorted_list = []
    expected_list = []

    for k in range(N_frames):
        # Measured (distorted) C markers from calibration readings
        c_measured = cal_readings['frames'][k]['c_points']

        # Expected (true) C markers from output1
        c_expected = output1_data['c_expected_frames'][k]

        distorted_list.append(c_measured)
        expected_list.append(c_expected)

    # Stack all frames into single arrays
    distorted_points = np.vstack(distorted_list)
    expected_points = np.vstack(expected_list)

    return distorted_points, expected_points


def read_em_marker_data(file_path):
    """
    Read EM marker data file (used for fiducials, navigation, etc.).

    This is a generic reader for files with the format:
    - First line: N_G, N_frames
    - Remaining lines: marker positions (N_frames * N_G rows of 3D coordinates)

    Args:
        file_path (str): Path to the EM marker data file

    Returns:
        dict: Dictionary containing:
            - 'N_G': Number of markers on probe
            - 'N_frames': Number of frames
            - 'frames': List of arrays with probe marker positions per frame
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split(',')
            N_G = int(first_line[0])
            N_frames = int(first_line[1])
            data = np.loadtxt(file, delimiter=',')

        # Reshape into frames
        frames = [data[i*N_G:(i+1)*N_G] for i in range(N_frames)]

        return {
            'N_G': N_G,
            'N_frames': N_frames,
            'frames': frames
        }
    except Exception as e:
        print(f"Error reading EM marker data from {file_path}: {e}")
        return None


def read_em_fiducials(file_path):
    """Read EM fiducials data file (alias for read_em_marker_data)."""
    return read_em_marker_data(file_path)


def read_em_nav(file_path):
    """Read EM navigation data file (alias for read_em_marker_data)."""
    return read_em_marker_data(file_path)


def read_ct_fiducials(file_path):
    """
    Read CT fiducials data file.

    Args:
        file_path (str): Path to the CT fiducials file

    Returns:
        dict: Dictionary containing:
            - 'N_B': Number of fiducials
            - 'b_points': Array of fiducial positions in CT coordinates, shape (N_B, 3)
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split(',')
            N_B = int(first_line[0])
            b_points = np.loadtxt(file, delimiter=',')

        return {
            'N_B': N_B,
            'b_points': b_points
        }
    except Exception as e:
        print(f"Error reading CT fiducials file from {file_path}: {e}")
        return None


def compute_em_fiducials(em_fiducials_file, p_tip, correction_model, probe_local):
    """
    Compute fiducial positions in EM tracker base coordinates (Step 4 of PA2).

    This function:
    1. Reads EM fiducials data (probe touching each fiducial)
    2. Applies distortion correction to probe markers
    3. Uses the SAME probe local frame from pivot calibration (critical for consistency!)
    4. Transforms calibrated tip position to EM base coordinates

    Args:
        em_fiducials_file (str): Path to EM fiducials file (NAME-em-fiducialss.txt)
        p_tip (np.ndarray): Calibrated probe tip position in probe local frame, shape (3,)
        correction_model (dict): Fitted distortion correction model
        probe_local (np.ndarray): Probe local frame from pivot calibration, shape (N_G, 3)
                                  MUST be the same frame used to calibrate p_tip!

    Returns:
        np.ndarray: Fiducial positions in EM base coordinates, shape (N_B, 3)
    """

    # Read EM fiducials data
    em_fiducials = read_em_fiducials(em_fiducials_file)
    if em_fiducials is None:
        return None

    # Apply distortion correction to all frames first
    corrected_fiducial_frames = []
    for frame in em_fiducials['frames']:
        corrected_frame = distortion_correction.correct_frame_markers(frame, correction_model)
        corrected_fiducial_frames.append(corrected_frame)

    # Compute fiducial positions in EM base coordinates
    # CRITICAL: Use the probe_local from pivot calibration, NOT from fiducial frames!
    b_em = []
    for corrected_markers in corrected_fiducial_frames:
        # Use point set registration with probe_local from pivot calibration
        probe_transform = FrameTransform.Point_set_registration(probe_local, corrected_markers)

        # Transform tip position to EM base coordinates
        fiducial_em = probe_transform.transform_points(p_tip.reshape(1, -1))[0]
        b_em.append(fiducial_em)

    return np.array(b_em)


def em_pivot_corrected(empivot_file, calreadings_file, output1_file, output_file=None, output_dir="output", degree=5):
    """
    Perform EM pivot calibration with distortion correction.

    This function:
    1. Reads input files (NAME-empivot.txt, NAME-calreadings.txt, NAME-output1.txt)
    2. Fits distortion correction using calibration data
    3. Applies correction to EM pivot data
    4. Performs pivot calibration with corrected data
    5. Optionally saves results if output_file is provided

    Args:
        empivot_file (str): Path to EM pivot data file (NAME-empivot.txt)
        calreadings_file (str): Path to calibration readings file (NAME-calreadings.txt)
        output1_file (str): Path to output1 file (NAME-output1.txt)
        output_file (str, optional): Output file name (if None, doesn't save to file)
        output_dir (str): Output directory path
        degree (int): Polynomial degree for distortion correction (default: 5)

    Returns:
        dict: Dictionary containing:
            - 'correction_model': Fitted distortion correction model
            - 'tip_position': Calibrated tip position (p_tip)
            - 'pivot_point': Calibrated pivot point (p_pivot)
            - 'residual_error': RMS residual error from calibration
    """
    # Read input data
    empivot = read_pivot_data(empivot_file)
    cal_readings = read_cal_data(calreadings_file)
    output1_data = read_output1(output1_file)

    # Extract paired data for distortion calibration
    distorted_points, expected_points = extract_distortion_calibration_data(
        cal_readings, output1_data
    )

    # Fit distortion correction model
    correction_model = distortion_correction.fit_distortion_correction(
        distorted_points, expected_points, degree=degree
    )

    # Apply distortion correction to all EM pivot frames
    corrected_frames = []
    for frame in empivot['frames']:
        corrected_frame = distortion_correction.correct_frame_markers(frame, correction_model)
        corrected_frames.append(corrected_frame)

    # Create corrected empivot data structure
    empivot_corrected = {
        'frames': corrected_frames,
        'N_G': empivot['N_G']
    }

    # Perform pivot calibration with corrected data
    # IMPORTANT: Also compute probe_local frame used in calibration for consistency
    probe_local = pivot_calibration.first_frame_centroid(corrected_frames)
    result = pivot_calibration.em_pivot_calibration(empivot_corrected)
    p_tip = result['tip_position']
    p_pivot = result['pivot_point']
    residual_error = result['residual_error']

    # Return results including probe_local for consistent downstream usage
    return {
        'correction_model': correction_model,
        'tip_position': p_tip,
        'pivot_point': p_pivot,
        'residual_error': residual_error,
        'probe_local': probe_local
    }

def pa2_output2(empivot_file, em_fiducials_file, ct_fiducials_file, em_nav_file,
                calreadings_file, output1_file, output_file, output_dir="output", degree=5):
    """
    Complete PA2 pipeline: Generate output2 file with probe tip positions in CT coordinates.

    This function performs the complete PA2 workflow (Steps 3-6):
    1. Perform EM pivot calibration with distortion correction (Step 3)
    2. Compute fiducial positions in EM coordinates (Step 4)
    3. Compute F_reg registration (EM → CT) (Step 5)
    4. Apply to navigation data to get tip positions in CT (Step 6)
    5. Write output2 file

    Args:
        empivot_file (str): Path to EM pivot file (NAME-empivot.txt)
        em_fiducials_file (str): Path to EM fiducials file (NAME-em-fiducialss.txt)
        ct_fiducials_file (str): Path to CT fiducials file (NAME-ct-fiducials.txt)
        em_nav_file (str): Path to EM navigation file (NAME-EM-nav.txt)
        calreadings_file (str): Path to calibration readings file (NAME-calreadings.txt)
        output1_file (str): Path to output1 file (NAME-output1.txt)
        output_file (str): Output file name (e.g., "pa2-debug-a-output2")
        output_dir (str): Output directory path
        degree (int): Polynomial degree for distortion correction (default: 5)
    """
    # Step 3: Perform EM pivot calibration with distortion correction
    # CRITICAL: Get probe_local from pivot calibration for consistency
    pivot_result = em_pivot_corrected(
        empivot_file,
        calreadings_file,
        output1_file,
        output_dir=output_dir,
        degree=degree
    )
    correction_model = pivot_result['correction_model']
    p_tip = pivot_result['tip_position']
    probe_local = pivot_result['probe_local']

    # Step 4: Compute fiducial positions in EM coordinates
    # CRITICAL: Use the SAME probe_local from pivot calibration!
    b_em = compute_em_fiducials(em_fiducials_file, p_tip, correction_model, probe_local)

    # Step 5: Compute F_reg
    ct_fiducials = read_ct_fiducials(ct_fiducials_file)
    b_ct = ct_fiducials['b_points']
    F_reg = FrameTransform.Point_set_registration(b_em, b_ct)

    # Step 6: Apply to navigation data
    em_nav = read_em_nav(em_nav_file)
    tip_ct_all = []

    for frame in em_nav['frames']:
        # Apply distortion correction
        corrected_markers = distortion_correction.correct_frame_markers(frame, correction_model)

        # Find probe pose in EM base using the same probe_local frame
        probe_transform = FrameTransform.Point_set_registration(probe_local, corrected_markers)

        # Transform tip to EM base
        tip_em = probe_transform.transform_points(p_tip.reshape(1, -1))[0]

        # Transform tip to CT
        tip_ct = F_reg.transform_points(tip_em.reshape(1, -1))[0]
        tip_ct_all.append(tip_ct)

    # Write output2 file
    with open(f"{output_dir}/{output_file}.txt", 'w') as f:
        # Header: N_frames, filename
        f.write(f"{len(tip_ct_all)}, {output_file}.txt\n")

        # Write each tip position in CT coordinates
        for tip in tip_ct_all:
            f.write(f"{tip[0]:8.2f}, {tip[1]:8.2f}, {tip[2]:8.2f}\n")


# ========== PA3 Functions ==========

def read_body_markers_and_tip(file_path):
    """
    Read rigid body definition file (Problem3-BodyA or Problem3-BodyB).

    File format:
    - First line: N_markers ProblemX-BodyY.txt
    - Next N_markers lines: xyz coordinates of LED markers in body coordinates
    - Next line: xyz coordinates of tip in body coordinates

    Args:
        file_path (str): Path to the body definition file

    Returns:
        dict: Dictionary containing:
            - 'N_markers': Number of LED markers
            - 'markers': Array of marker positions in body coordinates, shape (N_markers, 3)
            - 'tip': Tip position in body coordinates, shape (3,)
            - 'name': Body name (e.g., "Problem3-BodyA")
    """
    try:
        with open(file_path, 'r') as file:
            # Read header line: "6 Problem3-BodyA.txt"
            first_line = file.readline().strip().split()
            N_markers = int(first_line[0])
            body_name = first_line[1] if len(first_line) > 1 else ""

            # Read marker positions (space-separated)
            markers = []
            for _ in range(N_markers):
                line = file.readline().strip().split()
                marker = [float(x) for x in line]
                markers.append(marker)
            markers = np.array(markers)

            # Read tip position (space-separated)
            tip_line = file.readline().strip().split()
            tip = np.array([float(x) for x in tip_line])

        return {
            'N_markers': N_markers,
            'markers': markers,
            'tip': tip,
            'name': body_name
        }
    except Exception as e:
        print(f"Error reading body definition file from {file_path}: {e}")
        return None


def read_surface_mesh(file_path):
    """
    Read surface mesh file (ProblemXMesh.sur).

    File format:
    - First line: N_vertices
    - Next N_vertices lines: xyz coordinates of vertices in CT coordinates
    - Next line: N_triangles
    - Next N_triangles lines: 6 integers per triangle
      (i1, i2, i3, n1, n2, n3) where:
      - i1, i2, i3 are vertex indices for the triangle
      - n1, n2, n3 are neighbor triangle indices (not needed for PA3)

    Args:
        file_path (str): Path to the surface mesh file

    Returns:
        dict: Dictionary containing:
            - 'N_vertices': Number of vertices
            - 'vertices': Array of vertex positions in CT coordinates, shape (N_vertices, 3)
            - 'N_triangles': Number of triangles
            - 'triangles': Array of triangle vertex indices, shape (N_triangles, 3)
            - 'neighbors': Array of neighbor triangle indices, shape (N_triangles, 3)
    """
    try:
        with open(file_path, 'r') as file:
            # Read number of vertices
            N_vertices = int(file.readline().strip())

            # Read vertex coordinates (space-separated)
            vertices = []
            for _ in range(N_vertices):
                vertex = [float(x) for x in file.readline().strip().split()]
                vertices.append(vertex)
            vertices = np.array(vertices)

            # Read number of triangles
            N_triangles = int(file.readline().strip())

            # Read triangle indices and neighbors (space-separated)
            triangles = []
            neighbors = []
            for _ in range(N_triangles):
                indices = [int(x) for x in file.readline().strip().split()]
                triangles.append(indices[:3])  # First 3 are vertex indices
                neighbors.append(indices[3:])  # Last 3 are neighbor indices
            triangles = np.array(triangles)
            neighbors = np.array(neighbors)

        return {
            'N_vertices': N_vertices,
            'vertices': vertices,
            'N_triangles': N_triangles,
            'triangles': triangles,
            'neighbors': neighbors
        }
    except Exception as e:
        print(f"Error reading surface mesh file from {file_path}: {e}")
        return None


def read_sample_readings(file_path):
    """
    Read sample readings file (PA3-X-Debug/Unknown-SampleReadingsTest.txt).

    File format:
    - First line: N_s, N_samps, filename 0  (where N_s = N_A + N_B + N_D)
    - For each sample (repeated N_samps times):
      - Next N_A lines: xyz coordinates of A body LED markers in tracker coordinates
      - Next N_B lines: xyz coordinates of B body LED markers in tracker coordinates
      - Next N_D lines: xyz coordinates of dummy markers in tracker coordinates (if any)

    Args:
        file_path (str): Path to the sample readings file

    Returns:
        dict: Dictionary containing:
            - 'N_s': Total number of markers per frame
            - 'N_samps': Number of sample frames
            - 'filename': Original filename
            - 'data': Raw marker data array, shape (N_s * N_samps, 3)
    """
    try:
        with open(file_path, 'r') as file:
            # Read header line: "16, 15, PA3-A-Debug-SampleReadingsTest.txt 0"
            first_line = file.readline().strip()
            # Split by comma first
            parts = first_line.split(',')
            N_s = int(parts[0].strip())
            N_samps = int(parts[1].strip())
            # The filename might have spaces and a trailing 0
            filename_part = parts[2].strip() if len(parts) > 2 else ""
            filename = filename_part.split()[0] if filename_part else ""

            # Read all marker data
            data = np.loadtxt(file, delimiter=',')

        return {
            'N_s': N_s,
            'N_samps': N_samps,
            'filename': filename,
            'data': data  # Raw data to be parsed with N_A, N_B info
        }
    except Exception as e:
        print(f"Error reading sample readings file from {file_path}: {e}")
        return None


def parse_sample_readings(sample_data, N_A, N_B):
    """
    Parse raw sample readings data into structured frames.

    Args:
        sample_data (dict): Dictionary from read_sample_readings()
        N_A (int): Number of A body markers
        N_B (int): Number of B body markers

    Returns:
        dict: Dictionary containing:
            - 'N_A': Number of A body markers
            - 'N_B': Number of B body markers
            - 'N_D': Number of dummy markers
            - 'N_samps': Number of sample frames
            - 'frames': List of dictionaries, each containing:
                - 'a_markers': Array of A body marker positions, shape (N_A, 3)
                - 'b_markers': Array of B body marker positions, shape (N_B, 3)
                - 'd_markers': Array of dummy marker positions, shape (N_D, 3) (if any)
    """
    N_s = sample_data['N_s']
    N_samps = sample_data['N_samps']
    data = sample_data['data']

    N_D = N_s - N_A - N_B

    frames = []
    for k in range(N_samps):
        start_idx = k * N_s
        end_idx = (k + 1) * N_s

        frame_data = data[start_idx:end_idx]

        a_markers = frame_data[:N_A]
        b_markers = frame_data[N_A:N_A+N_B]
        d_markers = frame_data[N_A+N_B:] if N_D > 0 else np.array([])

        frames.append({
            'a_markers': a_markers,
            'b_markers': b_markers,
            'd_markers': d_markers
        })

    return {
        'N_A': N_A,
        'N_B': N_B,
        'N_D': N_D,
        'N_samps': N_samps,
        'frames': frames
    }


def write_pa3_output(output_file, results, output_dir="output"):
    """
    Write PA3 output file in the required format.

    Output format:
    - First line: N_samps "filename" 0
    - For each sample: d_x d_y d_z c_x c_y c_z ||d_k - c_k||

    Args:
        output_file (str): Output file name (e.g., "PA3-A-Debug-Output")
        results (list): List of result dictionaries, each containing:
            - 'd_k': Pointer tip position in bone frame, shape (3,)
            - 'c_k': Closest point on mesh, shape (3,)
            - 'distance': Distance between d_k and c_k
        output_dir (str): Output directory path
    """
    N_samps = len(results)

    with open(f"{output_dir}/{output_file}.txt", 'w') as f:
        # Header line: N_samps filename 0
        f.write(f"{N_samps} {output_file}.txt 0\n")

        # Write each sample result
        for result in results:
            d_k = result['d_k']
            c_k = result['c_k']
            distance = result['distance']

            # Format: d_x d_y d_z c_x c_y c_z ||d_k - c_k||
            # Use spaces (not commas) to match expected output format
            f.write(f"{d_k[0]:8.2f} {d_k[1]:8.2f} {d_k[2]:8.2f}     "
                   f"{c_k[0]:8.2f} {c_k[1]:8.2f} {c_k[2]:8.2f}  "
                   f"{distance:8.3f}\n")


def write_pa4_output(output_file, results, output_dir="output"):
    """
    Write PA4 output file in the required format.

    Output format:
    - First line: N_samps "filename" 0
    - For each sample: s_x s_y s_z c_x c_y c_z ||s_k - c_k||

    Args:
        output_file (str): Output file name (e.g., "PA4-A-Debug-Output")
        results (list): List of result dictionaries, each containing:
            - 's_k': Registered pointer tip position (F_reg * d_k), shape (3,)
            - 'c_k': Closest point on mesh, shape (3,)
            - 'distance': Distance between s_k and c_k
        output_dir (str): Output directory path
    """
    N_samps = len(results)

    with open(f"{output_dir}/{output_file}.txt", 'w') as f:
        # Header line: N_samps filename 0
        f.write(f"{N_samps} {output_file}.txt 0\n")

        # Write each sample result
        for result in results:
            s_k = result['s_k']  # Note: PA4 outputs s_k (not d_k)
            c_k = result['c_k']
            distance = result['distance']

            # Format: s_x s_y s_z c_x c_y c_z ||s_k - c_k||
            # Use spaces (not commas) to match expected output format
            f.write(f"{s_k[0]:8.2f} {s_k[1]:8.2f} {s_k[2]:8.2f}     "
                   f"{c_k[0]:8.2f} {c_k[1]:8.2f} {c_k[2]:8.2f}  "
                   f"{distance:8.3f}\n")



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
        self.read_output1 = read_output1
        self.extract_distortion_calibration_data = extract_distortion_calibration_data
        self.em_pivot_corrected = em_pivot_corrected
        self.read_em_marker_data = read_em_marker_data
        self.read_em_fiducials = read_em_fiducials
        self.read_em_nav = read_em_nav
        self.read_ct_fiducials = read_ct_fiducials
        self.compute_em_fiducials = compute_em_fiducials
        self.pa2_output2 = pa2_output2
        # PA3 functions
        self.read_body_markers_and_tip = read_body_markers_and_tip
        self.read_surface_mesh = read_surface_mesh
        self.read_sample_readings = read_sample_readings
        self.parse_sample_readings = parse_sample_readings
        self.write_pa3_output = write_pa3_output
        # PA4 functions
        self.write_pa4_output = write_pa4_output


# Create an instance to be imported
utility_functions = UtilityFunctions()
