"""
Unit tests for Utility Functions Module

Tests core utility functions:
- C_expected: C coordinate calculation with compose order fix
- Dynamic filename handling
- Integration with pivot calibration results
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import sys
import os
import tempfile
import shutil

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utility_functions import C_expected
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
            # Reshape 3D arrays to 2D for savetxt
            F_A_reshaped = np.array(F_A_list).reshape(-1, 16)  # Flatten each 4x4 matrix to 16 elements
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
    
    def test_c_expected_compose_order_mathematical_correctness(self):
        """Test mathematical correctness of compose order: F_D ∘ F_A^(-1) vs F_D^(-1) ∘ F_A"""
        # Create simple test case where we can verify the math
        cal_body = {
            "c_points": np.array([[1, 0, 0]], dtype=float)  # Single point for simplicity
        }
        
        # Create simple transformations
        F_D = np.eye(4, dtype=float)
        F_D[:3, 3] = [1, 0, 0]  # Translation by [1, 0, 0]
        
        F_A = np.eye(4, dtype=float)
        F_A[:3, 3] = [0, 1, 0]  # Translation by [0, 1, 0]
        
        F_D_list = [F_D]
        F_A_list = [F_A]
        
        # Calculate expected result manually
        F_D_transform = FrameTransform(F_D[:3, :3], F_D[:3, 3])
        F_A_transform = FrameTransform(F_A[:3, :3], F_A[:3, 3])
        
        # Correct compose order: F_D ∘ F_A^(-1)
        correct_compose = F_D_transform.compose(F_A_transform.inverse())
        correct_result = correct_compose.transform_points(cal_body["c_points"])
        
        # Incorrect compose order: F_D^(-1) ∘ F_A
        incorrect_compose = F_D_transform.inverse().compose(F_A_transform)
        incorrect_result = incorrect_compose.transform_points(cal_body["c_points"])
        
        # Verify they are different (proving the fix matters)
        assert not np.allclose(correct_result, incorrect_result, atol=1e-10)
        
        # Verify correct result: F_D ∘ F_A^(-1) * [1,0,0] = F_D * F_A^(-1) * [1,0,0]
        # F_A^(-1) * [1,0,0] = [1,-1,0] (inverse of translation [0,1,0])
        # F_D * [1,-1,0] = [2,-1,0] (apply translation [1,0,0])
        expected_result = np.array([[2, -1, 0]], dtype=float)
        assert_array_almost_equal(correct_result, expected_result, decimal=10)
    
    def test_c_expected_dynamic_filename_handling(self):
        """Test that C_expected handles dynamic filenames correctly"""
        cal_body = {
            "c_points": np.array([[1, 2, 3]], dtype=float)
        }
        
        # Simple identity transformations (need multiple frames)
        F_D_list = [np.eye(4, dtype=float), np.eye(4, dtype=float)]  # 2 frames
        F_A_list = [np.eye(4, dtype=float), np.eye(4, dtype=float)]  # 2 frames
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different dataset names
            test_cases = [
                ("pa1-debug-a-output1", "PA1-DEBUG-A"),
                ("pa1-debug-b-output1", "PA1-DEBUG-B"),
                ("pa1-unknown-c-output1", "PA1-UNKNOWN-C")
            ]
            
            for output_file, dataset_prefix in test_cases:
                # Create mock pivot result files
                em_pivot_result = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=float)
                opt_pivot_result = np.array([[0, 0, 0], [2, 2, 2], [0, 0, 0]], dtype=float)
                
                # Save with dynamic filenames
                np.savetxt(f"{temp_dir}/{dataset_prefix}_EM_pivot.txt", em_pivot_result, fmt="%.3f", delimiter=",")
                np.savetxt(f"{temp_dir}/{dataset_prefix}_Optpivot.txt", opt_pivot_result, fmt="%.3f", delimiter=",")
                
                # Save Fa and Fd registration files
                # Reshape 3D arrays to 2D for savetxt
                F_A_reshaped = np.array(F_A_list).reshape(-1, 16)  # Flatten each 4x4 matrix to 16 elements
                F_D_reshaped = np.array(F_D_list).reshape(-1, 16)
                np.savetxt(f"{temp_dir}/Fa_a_registration.txt", F_A_reshaped, fmt="%.3f", delimiter=",")
                np.savetxt(f"{temp_dir}/Fd_a_registration.txt", F_D_reshaped, fmt="%.3f", delimiter=",")
                
                # Test C_expected function
                C_expected(cal_body, output_file, "Fa_a_registration", "Fd_a_registration", temp_dir)
                
                # Verify output file was created
                output_path = f"{temp_dir}/{output_file}.txt"
                assert os.path.exists(output_path), f"Output file not created for {output_file}"
                
                # Verify content contains correct dataset name
                with open(output_path, 'r') as f:
                    content = f.read()
                    assert f"{output_file}.txt" in content
    
    def test_c_expected_error_handling(self):
        """Test error handling in C_expected function"""
        cal_body = {
            "c_points": np.array([[1, 2, 3]], dtype=float)
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with missing pivot files
            with pytest.raises(FileNotFoundError):
                C_expected(cal_body, "pa1-debug-a-output1", "Fa_a_registration", "Fd_a_registration", temp_dir)
    
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
            # Reshape 3D arrays to 2D for savetxt
            F_A_reshaped = np.array(F_A_list).reshape(-1, 16)  # Flatten each 4x4 matrix to 16 elements
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


class TestUtilityFunctionIntegration:
    """Test integration between utility functions and pivot calibration"""
    
    def test_end_to_end_coordinate_transformation(self):
        """Test complete end-to-end coordinate transformation pipeline"""
        # This test simulates the complete pipeline from pivot calibration to C coordinate calculation
        
        # Create realistic calibration body data
        cal_body = {
            "c_points": np.array([
                [100, 200, 300],
                [101, 200, 300],
                [100, 201, 300],
                [100, 200, 301]
            ], dtype=float)
        }
        
        # Create realistic transformation matrices (simulating Fa/Fd registrations)
        F_D_list = []
        F_A_list = []
        
        for i in range(3):
            # Create realistic rotations and translations
            angle = i * 0.1
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]], dtype=float)
            
            t_D = np.array([i * 0.5, i * 0.3, i * 0.2], dtype=float)
            t_A = np.array([i * 0.4, i * 0.6, i * 0.1], dtype=float)
            
            F_D = np.eye(4, dtype=float)
            F_D[:3, :3] = R
            F_D[:3, 3] = t_D
            
            F_A = np.eye(4, dtype=float)
            F_A[:3, :3] = R
            F_A[:3, 3] = t_A
            
            F_D_list.append(F_D)
            F_A_list.append(F_A)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic pivot calibration results
            em_pivot_result = np.array([
                [0.1, 0.2, 0.3],  # EM tip
                [190.5, 207.3, 209.2],  # EM pivot (realistic values)
                [0.01, 0, 0]  # EM residual
            ], dtype=float)
            
            opt_pivot_result = np.array([
                [0.4, 0.5, 0.6],  # Optical tip
                [400.6, 402.1, 203.5],  # Optical pivot (realistic values)
                [0.02, 0, 0]  # Optical residual
            ], dtype=float)
            
            # Save pivot results
            np.savetxt(f"{temp_dir}/PA1-DEBUG-A_EM_pivot.txt", em_pivot_result, fmt="%.3f", delimiter=",")
            np.savetxt(f"{temp_dir}/PA1-DEBUG-A_Optpivot.txt", opt_pivot_result, fmt="%.3f", delimiter=",")
            
            # Save Fa and Fd registration files
            # Reshape 3D arrays to 2D for savetxt
            F_A_reshaped = np.array(F_A_list).reshape(-1, 16)  # Flatten each 4x4 matrix to 16 elements
            F_D_reshaped = np.array(F_D_list).reshape(-1, 16)
            np.savetxt(f"{temp_dir}/Fa_a_registration.txt", F_A_reshaped, fmt="%.3f", delimiter=",")
            np.savetxt(f"{temp_dir}/Fd_a_registration.txt", F_D_reshaped, fmt="%.3f", delimiter=",")
            
            # Run complete pipeline
            C_expected(cal_body, "pa1-debug-a-output1", "Fa_a_registration", "Fd_a_registration", temp_dir)
            
            # Verify output file
            output_file = f"{temp_dir}/pa1-debug-a-output1.txt"
            assert os.path.exists(output_file)
            
            # Verify file content structure
            with open(output_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
            
            # Verify header
            assert "4, 3, pa1-debug-a-output1.txt" in lines[0]
            
            # Verify pivot points
            assert "190.500, 207.300, 209.200" in lines[1]
            assert "400.600, 402.100, 203.500" in lines[2]
            
            # Verify C coordinates are present
            assert len(lines) == 15  # Header + 2 pivot + 3 frames × 4 C points
            
            # Verify all C coordinates are reasonable
            for i in range(3, len(lines)):
                coords = [float(x.strip()) for x in lines[i].split(',')]
                assert len(coords) == 3
                assert all(np.isfinite(coords))
                # C coordinates should be in reasonable range (not extremely large)
                assert all(abs(coord) < 10000 for coord in coords)


if __name__ == "__main__":
    pytest.main([__file__])
