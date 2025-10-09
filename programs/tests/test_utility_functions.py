"""
Unit tests for Utility Functions Module

Tests core utility functions using synthetic data:
- C_expected: C coordinate calculation with compose order fix
- File I/O functions for data reading
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys
import os
import tempfile

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utility_functions import C_expected, read_cal_data, read_pivot_data, read_optpivot
from frame_transform import FrameTransform


class TestCExpectedFunction:
    """Test cases for C_expected function with compose order fix"""
    
    def test_c_expected_compose_order_correctness(self):
        """Test that C_expected uses correct compose order: F_D ∘ F_A^(-1)"""
        # Create test data
        cal_body = {
            "c_points": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        }
        
        # Create test transformations
        F_D_list = []
        F_A_list = []
        
        for i in range(3):
            # Create rotation matrices
            angle = i * np.pi / 4
            R_D = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]], dtype=float)
            R_A = np.array([[np.cos(angle + np.pi/6), -np.sin(angle + np.pi/6), 0],
                           [np.sin(angle + np.pi/6), np.cos(angle + np.pi/6), 0],
                           [0, 0, 1]], dtype=float)
            
            # Create translation vectors
            t_D = np.array([i * 0.1, i * 0.2, i * 0.3], dtype=float)
            t_A = np.array([i * 0.2, i * 0.3, i * 0.4], dtype=float)
            
            # Create 4x4 transformation matrices
            F_D = np.eye(4, dtype=float)
            F_D[:3, :3] = R_D
            F_D[:3, 3] = t_D
            
            F_A = np.eye(4, dtype=float)
            F_A[:3, :3] = R_A
            F_A[:3, 3] = t_A
            
            F_D_list.append(F_D)
            F_A_list.append(F_A)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock pivot result files
            em_pivot_result = np.array([
                [0.1, 0.2, 0.3],  # tip position
                [1.0, 2.0, 3.0],  # pivot point
                [0.01, 0, 0]       # residual error
            ], dtype=float)
            
            opt_pivot_result = np.array([
                [0.4, 0.5, 0.6],  # tip position
                [4.0, 5.0, 6.0],  # pivot point
                [0.02, 0, 0]       # residual error
            ], dtype=float)
            
            # Save mock files
            np.savetxt(f"{temp_dir}/PA1-DEBUG-A_EM_pivot.txt", em_pivot_result, fmt="%.3f", delimiter=",")
            np.savetxt(f"{temp_dir}/PA1-DEBUG-A_Optpivot.txt", opt_pivot_result, fmt="%.3f", delimiter=",")
            
            # Save Fa and Fd registration files
            F_A_reshaped = np.array(F_A_list).reshape(-1, 16)
            F_D_reshaped = np.array(F_D_list).reshape(-1, 16)
            np.savetxt(f"{temp_dir}/Fa_a_registration.txt", F_A_reshaped, fmt="%.3f", delimiter=",")
            np.savetxt(f"{temp_dir}/Fd_a_registration.txt", F_D_reshaped, fmt="%.3f", delimiter=",")
            
            # Test C_expected function
            C_expected(cal_body, "pa1-debug-a-output1", "Fa_a_registration", "Fd_a_registration", temp_dir)
            
            # Verify output file was created
            output_file = f"{temp_dir}/pa1-debug-a-output1.txt"
            assert os.path.exists(output_file)
            
            # Read and verify output file content
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Check header line
            assert "3, 3, pa1-debug-a-output1.txt" in lines[0]
            
            # Check EM pivot tip position
            em_tip_line = lines[1].strip()
            assert "1.000, 2.000, 3.000" in em_tip_line
            
            # Check optical pivot tip position
            opt_tip_line = lines[2].strip()
            assert "4.000, 5.000, 6.000" in opt_tip_line
    
    def test_c_expected_coordinate_accuracy(self):
        """Test that C_expected produces accurate coordinate transformations"""
        # Create test data with known ground truth
        cal_body = {
            "c_points": np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=float)
        }
        
        # Create transformations that should produce predictable results
        F_D_list = []
        F_A_list = []
        
        for i in range(2):
            # Create rotation around Z-axis
            angle = i * np.pi / 2
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]], dtype=float)
            
            # Create translations
            t_D = np.array([i * 0.1, i * 0.2, i * 0.3], dtype=float)
            t_A = np.array([i * 0.2, i * 0.3, i * 0.4], dtype=float)
            
            # Create 4x4 matrices
            F_D = np.eye(4, dtype=float)
            F_D[:3, :3] = R
            F_D[:3, 3] = t_D
            
            F_A = np.eye(4, dtype=float)
            F_A[:3, :3] = R
            F_A[:3, 3] = t_A
            
            F_D_list.append(F_D)
            F_A_list.append(F_A)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock pivot files
            em_pivot_result = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=float)
            opt_pivot_result = np.array([[0, 0, 0], [2, 2, 2], [0, 0, 0]], dtype=float)
            
            np.savetxt(f"{temp_dir}/PA1-DEBUG-A_EM_pivot.txt", em_pivot_result, fmt="%.3f", delimiter=",")
            np.savetxt(f"{temp_dir}/PA1-DEBUG-A_Optpivot.txt", opt_pivot_result, fmt="%.3f", delimiter=",")
            
            # Save Fa and Fd registration files
            F_A_reshaped = np.array(F_A_list).reshape(-1, 16)
            F_D_reshaped = np.array(F_D_list).reshape(-1, 16)
            np.savetxt(f"{temp_dir}/Fa_a_registration.txt", F_A_reshaped, fmt="%.3f", delimiter=",")
            np.savetxt(f"{temp_dir}/Fd_a_registration.txt", F_D_reshaped, fmt="%.3f", delimiter=",")
            
            # Run C_expected
            C_expected(cal_body, "pa1-debug-a-output1", "Fa_a_registration", "Fd_a_registration", temp_dir)
            
            # Read output and verify structure
            output_file = f"{temp_dir}/pa1-debug-a-output1.txt"
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Verify header
            assert len(lines) >= 3  # Header + 2 pivot lines + C coordinates
            
            # Verify C coordinates are present and reasonable
            c_coord_lines = lines[3:]  # Skip header and pivot lines
            assert len(c_coord_lines) == 8  # 2 frames × 4 C points
            
            # Verify all C coordinates are finite numbers
            for line in c_coord_lines:
                coords = [float(x.strip()) for x in line.strip().split(',')]
                assert len(coords) == 3
                assert all(np.isfinite(coords))


class TestFileIOFunctions:
    """Test cases for file I/O utility functions"""
    
    def test_read_cal_data_synthetic(self):
        """Test reading calibration data with synthetic data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write synthetic calibration data
            f.write("3, 3, 3\n")  # N_D, N_A, N_C
            f.write("1.0, 2.0, 3.0\n")
            f.write("4.0, 5.0, 6.0\n")
            f.write("7.0, 8.0, 9.0\n")
            f.write("10.0, 11.0, 12.0\n")
            f.write("13.0, 14.0, 15.0\n")
            f.write("16.0, 17.0, 18.0\n")
            f.write("19.0, 20.0, 21.0\n")
            f.write("22.0, 23.0, 24.0\n")
            f.write("25.0, 26.0, 27.0\n")
            f.flush()
            
            # Test reading
            result = read_cal_data(f.name)
            
            # Verify structure
            assert result is not None
            assert "d_points" in result
            assert "a_points" in result
            assert "c_points" in result
            
            # Verify data shapes
            assert result["d_points"].shape == (3, 3)
            assert result["a_points"].shape == (3, 3)
            assert result["c_points"].shape == (3, 3)
            
            # Verify data content
            assert np.allclose(result["d_points"][0], [1.0, 2.0, 3.0])
            assert np.allclose(result["a_points"][0], [10.0, 11.0, 12.0])
            assert np.allclose(result["c_points"][0], [19.0, 20.0, 21.0])
        
        # Clean up
        os.unlink(f.name)
    
    def test_read_pivot_data_synthetic(self):
        """Test reading pivot data with synthetic data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write synthetic pivot data
            f.write("3, 2\n")  # N_G, N_frames
            f.write("1.0, 2.0, 3.0\n")
            f.write("4.0, 5.0, 6.0\n")
            f.write("7.0, 8.0, 9.0\n")
            f.write("10.0, 11.0, 12.0\n")
            f.write("13.0, 14.0, 15.0\n")
            f.write("16.0, 17.0, 18.0\n")
            f.flush()
            
            # Test reading
            result = read_pivot_data(f.name)
            
            # Verify structure
            assert result is not None
            assert "frames" in result
            assert "N_G" in result
            
            # Verify data shapes
            assert len(result["frames"]) == 2
            assert result["N_G"] == 3
            assert result["frames"][0].shape == (3, 3)
            assert result["frames"][1].shape == (3, 3)
            
            # Verify data content
            assert np.allclose(result["frames"][0][0], [1.0, 2.0, 3.0])
            assert np.allclose(result["frames"][1][0], [10.0, 11.0, 12.0])
        
        # Clean up
        os.unlink(f.name)
    
    def test_read_optpivot_synthetic(self):
        """Test reading optical pivot data with synthetic data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write synthetic optical pivot data
            f.write("2, 2, 2\n")  # N_D, N_H, N_frames
            f.write("1.0, 2.0, 3.0\n")
            f.write("4.0, 5.0, 6.0\n")
            f.write("7.0, 8.0, 9.0\n")
            f.write("10.0, 11.0, 12.0\n")
            f.write("13.0, 14.0, 15.0\n")
            f.write("16.0, 17.0, 18.0\n")
            f.write("19.0, 20.0, 21.0\n")
            f.write("22.0, 23.0, 24.0\n")
            f.flush()
            
            # Test reading
            result = read_optpivot(f.name)
            
            # Verify structure
            assert result is not None
            assert "frames" in result
            assert "N_D" in result
            assert "N_H" in result
            
            # Verify data shapes
            assert len(result["frames"]) == 2
            assert result["N_D"] == 2
            assert result["N_H"] == 2
            
            # Verify frame structure
            frame = result["frames"][0]
            assert "d_points" in frame
            assert "h_points" in frame
            assert frame["d_points"].shape == (2, 3)
            assert frame["h_points"].shape == (2, 3)
            
            # Verify data content
            assert np.allclose(frame["d_points"][0], [1.0, 2.0, 3.0])
            assert np.allclose(frame["h_points"][0], [7.0, 8.0, 9.0])
        
        # Clean up
        os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__])
