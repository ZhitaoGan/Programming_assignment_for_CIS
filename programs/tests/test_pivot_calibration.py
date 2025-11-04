"""
Unit tests for Pivot Calibration Module

Tests core pivot calibration algorithms using synthetic data:
- solve_for_pivot: Core least squares algorithm
- em_pivot_calibration: EM pivot calibration
- opt_pivot_calibration: Optical pivot calibration
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pivot_calibration import (
    solve_for_pivot, em_pivot_calibration, opt_pivot_calibration
)


class TestSolveForPivot:
    """Test cases for solve_for_pivot function - the core least squares algorithm"""
    
    def test_solve_for_pivot_ground_truth_validation(self):
        """Test solve_for_pivot with known ground truth - validates least squares accuracy"""
        # Create synthetic data with KNOWN tip and pivot positions
        true_tip = np.array([1.0, 2.0, 3.0])
        true_pivot = np.array([4.0, 5.0, 6.0])
        
        # Generate multiple frames with diverse rotations
        R_list = []
        p_list = []
        
        # Rotation around different axes
        angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        for angle in angles:
            # X-axis rotation
            R_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
            p_i = true_pivot - R_x @ true_tip
            R_list.append(R_x)
            p_list.append(p_i)
            
            # Y-axis rotation
            R_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
            p_i = true_pivot - R_y @ true_tip
            R_list.append(R_y)
            p_list.append(p_i)
        
        # Solve using least squares
        p_tip_result, p_pivot_result, residual_error = solve_for_pivot(R_list, p_list)
        
        # Validate that least squares recovered the ground truth
        assert_array_almost_equal(p_tip_result, true_tip, decimal=5)
        assert_array_almost_equal(p_pivot_result, true_pivot, decimal=5)
        assert residual_error < 1e-10
    
    def test_solve_for_pivot_with_noise(self):
        """Test solve_for_pivot robustness with synthetic noise"""
        # Known ground truth
        true_tip = np.array([1.0, 2.0, 3.0])
        true_pivot = np.array([4.0, 5.0, 6.0])
        
        # Generate frames with noise
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 8)
        R_list = []
        p_list = []
        
        for angle in angles:
            # Random rotation axis
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            
            # Correct equation: p_i = p_pivot - R_i * p_tip
            p_i_clean = true_pivot - R @ true_tip
            
            # Add noise to translation
            noise_level = 0.01  # 1% noise
            p_i_noisy = p_i_clean + noise_level * np.random.randn(3)
            
            R_list.append(R)
            p_list.append(p_i_noisy)
        
        # Solve with noisy data
        p_tip_result, p_pivot_result, residual_error = solve_for_pivot(R_list, p_list)
        
        # Should still recover ground truth within noise tolerance
        assert np.linalg.norm(p_tip_result - true_tip) < 0.1
        assert np.linalg.norm(p_pivot_result - true_pivot) < 0.1
        assert residual_error > 0


class TestEMPivotCalibration:
    """Test cases for EM pivot calibration algorithm"""
    
    def test_em_pivot_calibration_synthetic_data(self):
        """Test EM pivot calibration with synthetic data and ground truth validation"""
        # Known ground truth for pivot (in tracker frame)
        true_pivot = np.array([10.0, 20.0, 30.0])

        # Define probe local geometry (zero-mean as algorithm expects)
        probe_local_raw = np.array([
            [-0.3, -0.2, -0.1],
            [0.2, -0.2, -0.1],
            [-0.3, 0.1, -0.1],
            [-0.3, -0.2, 0.1]
        ], dtype=float)
        # Center the markers (algorithm does this automatically)
        probe_local = probe_local_raw - probe_local_raw.mean(axis=0)

        # Ground truth tip is relative to the centroid of markers
        true_tip = np.array([0.5, 0.3, 0.2])

        # Generate frames using pivot equation: pivot = R_i * tip + p_i
        # Rearranged: p_i = pivot - R_i * tip
        frames = []
        for i in range(8):
            # Rotate around different axes with different angles
            angle = i * np.pi / 4
            # Mix rotations for better conditioning
            if i % 3 == 0:
                R = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]])
            elif i % 3 == 1:
                R = np.array([[1, 0, 0],
                              [0, np.cos(angle), -np.sin(angle)],
                              [0, np.sin(angle), np.cos(angle)]])
            else:
                R = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])

            # Calculate translation using pivot equation
            p_i = true_pivot - R @ true_tip

            # Transform probe markers to this pose
            frame_markers = (R @ probe_local.T).T + p_i
            frames.append(frame_markers)

        result = em_pivot_calibration({"frames": frames})

        # Validate that algorithm recovers ground truth
        assert_array_almost_equal(result['tip_position'], true_tip, decimal=4)
        assert_array_almost_equal(result['pivot_point'], true_pivot, decimal=4)
        assert result['residual_error'] < 1e-8  # Should be near-zero for perfect data


class TestOptPivotCalibration:
    """Test cases for optical pivot calibration algorithm"""
    
    def test_opt_pivot_calibration_synthetic_data(self):
        """Test optical pivot calibration with synthetic data and ground truth validation"""
        # Known ground truth for pivot (in EM/tracker frame)
        true_pivot = np.array([15.0, 25.0, 35.0])

        # EM-base geometry (fixed markers on calibration body in EM coordinates)
        d_em = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)

        # Probe local geometry (markers in probe frame)
        h_local_raw = np.array([
            [-0.2, -0.15, -0.05],
            [0.2, -0.15, -0.05],
            [-0.2, 0.15, -0.05],
            [-0.2, -0.15, 0.05]
        ], dtype=float)
        # Center the markers (algorithm does this automatically)
        h_local = h_local_raw - h_local_raw.mean(axis=0)

        # Ground truth tip relative to centroid
        true_tip = np.array([0.4, 0.3, 0.1])

        # Generate frames
        frames = []
        for i in range(8):
            # Different probe poses (mix rotations for better conditioning)
            angle = i * np.pi / 6
            if i % 3 == 0:
                R_probe = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0],
                                   [0, 0, 1]])
            elif i % 3 == 1:
                R_probe = np.array([[1, 0, 0],
                                   [0, np.cos(angle), -np.sin(angle)],
                                   [0, np.sin(angle), np.cos(angle)]])
            else:
                R_probe = np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]])

            # Calculate probe translation using pivot equation: pivot = R * tip + p
            p_probe = true_pivot - R_probe @ true_tip

            # Transform probe markers to EM coordinates
            h_em = (R_probe @ h_local.T).T + p_probe

            # Simulate optical tracker measurements
            # Optical sees both D markers and H markers with some arbitrary transform
            # For simplicity, assume optical = EM (identity transform)
            d_optical = d_em.copy()
            h_optical = h_em.copy()

            frames.append({
                "d_points": d_optical,
                "h_points": h_optical
            })

        opt_pivot_data = {"frames": frames}
        cal_body_data = {"d_points": d_em}

        result = opt_pivot_calibration(opt_pivot_data, cal_body_data)

        # Validate that algorithm recovers ground truth
        assert_array_almost_equal(result['tip_position'], true_tip, decimal=4)
        assert_array_almost_equal(result['pivot_point'], true_pivot, decimal=4)
        assert result['residual_error'] < 1e-8  # Should be near-zero for perfect data


if __name__ == "__main__":
    pytest.main([__file__])