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
        """Test EM pivot calibration with synthetic data"""
        # Create synthetic EM pivot data with known ground truth
        true_tip = np.array([0.1, 0.2, 0.3])
        true_pivot = np.array([1.0, 2.0, 3.0])
        
        # Generate frames with known transformations
        frames = []
        for i in range(5):
            angle = i * np.pi / 4
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
            t = np.array([i * 0.1, i * 0.2, i * 0.3])
            
            # Create frame data (simulating EM measurements)
            frame_points = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], dtype=float)
            transformed_frame = (R @ frame_points.T).T + t
            frames.append(transformed_frame)
        
        result = em_pivot_calibration({"frames": frames})
        
        # Verify results structure
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        assert len(result['tip_position']) == 3
        assert len(result['pivot_point']) == 3
        assert result['residual_error'] >= 0
    
    def test_em_pivot_calibration_insufficient_frames(self):
        """Test EM pivot calibration with insufficient frames"""
        frames = [np.array([[0, 0, 0], [1, 0, 0]])]
        
        with pytest.raises(ValueError, match="EM pivot data must contain at least 2 frames"):
            em_pivot_calibration({"frames": frames})


class TestOptPivotCalibration:
    """Test cases for optical pivot calibration algorithm"""
    
    def test_opt_pivot_calibration_synthetic_data(self):
        """Test optical pivot calibration with synthetic data"""
        # Create synthetic optical pivot data
        frames = []
        for i in range(3):
            frames.append({
                "h_points": np.array([[i * 0.1, i * 0.2, i * 0.3],
                                    [i * 0.1 + 0.5, i * 0.2, i * 0.3],
                                    [i * 0.1, i * 0.2 + 0.5, i * 0.3]], dtype=float),
                "d_points": np.array([[10 + i * 0.1, 10 + i * 0.2, 10 + i * 0.3],
                                    [11 + i * 0.1, 10 + i * 0.2, 10 + i * 0.3],
                                    [10 + i * 0.1, 11 + i * 0.2, 10 + i * 0.3]], dtype=float)
            })
        
        opt_pivot_data = {"frames": frames}
        cal_body_data = {"d_points": frames[0]["d_points"]}
        
        result = opt_pivot_calibration(opt_pivot_data, cal_body_data)
        
        # Verify results structure
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        assert len(result['tip_position']) == 3
        assert len(result['pivot_point']) == 3
        assert result['residual_error'] >= 0
        assert np.all(np.isfinite(result['tip_position']))
        assert np.all(np.isfinite(result['pivot_point']))
    
    def test_opt_pivot_calibration_insufficient_frames(self):
        """Test optical pivot calibration with insufficient frames"""
        frames = [{"h_points": np.array([[0, 0, 0], [1, 0, 0]]),
                  "d_points": np.array([[10, 10, 10], [11, 10, 10]])}]
        opt_pivot_data = {"frames": frames}
        
        with pytest.raises(AssertionError):
            opt_pivot_calibration(opt_pivot_data)


if __name__ == "__main__":
    pytest.main([__file__])