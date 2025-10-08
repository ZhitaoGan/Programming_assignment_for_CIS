"""
Unit tests for Pivot Calibration Module

Tests core pivot calibration algorithms:
- solve_for_pivot: Core least squares algorithm
- em_pivot_calibration: EM pivot calibration
- opt_pivot_calibration: Optical pivot calibration
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pivot_calibration import (
    solve_for_pivot, em_pivot_calibration, opt_pivot_calibration
)


class TestSolveForPivot:
    """Test cases for solve_for_pivot function - the core least squares algorithm"""
    
    def test_solve_for_pivot_simple_case(self):
        """Test solve_for_pivot with simple known case"""
        # Create a simple case where we know the solution
        # Tool tip at origin, pivot at [1, 0, 0]
        R_list = [np.eye(3)]  # No rotation
        p_list = [np.array([0, 0, 0])]  # Tool tip at origin
        
        p_tip, p_pivot, residual_error = solve_for_pivot(R_list, p_list)
        
        # With only one frame, we can't determine both tip and pivot uniquely
        # But we can test that the function runs without error
        assert len(p_tip) == 3
        assert len(p_pivot) == 3
        assert isinstance(residual_error, (int, float))
    
    def test_solve_for_pivot_multiple_frames(self):
        """Test solve_for_pivot with multiple frames"""
        # Create frames with known tip and pivot
        tip = np.array([0, 0, 0])
        pivot = np.array([1, 0, 0])
        
        # Create rotations around Z-axis
        angles = [0, np.pi/4, np.pi/2]
        R_list = []
        p_list = []
        
        for angle in angles:
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
            # For each rotation, the tip position is: R * tip + t = R * pivot
            # So: t = R * pivot - R * tip = R * (pivot - tip)
            t = R @ (pivot - tip)
            
            R_list.append(R)
            p_list.append(t)
        
        p_tip_result, p_pivot_result, residual_error = solve_for_pivot(R_list, p_list)
        
        # Check that we get reasonable results
        assert len(p_tip_result) == 3
        assert len(p_pivot_result) == 3
        assert residual_error >= 0  # Residual should be non-negative


class TestEMPivotCalibration:
    """Test cases for EM pivot calibration algorithm"""
    
    def test_em_pivot_calibration_valid_input(self):
        """Test EM pivot calibration with valid input"""
        # Create test data with multiple frames
        frames = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
            np.array([[0.2, 0.2, 0.2], [1.2, 0.2, 0.2], [0.2, 1.2, 0.2]])
        ]
        
        em_pivot_data = {"frames": frames}
        
        result = em_pivot_calibration(em_pivot_data)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        
        assert len(result['tip_position']) == 3
        assert len(result['pivot_point']) == 3
        assert isinstance(result['residual_error'], (int, float))
    
    def test_em_pivot_calibration_insufficient_frames(self):
        """Test EM pivot calibration with insufficient frames"""
        frames = [np.array([[0, 0, 0], [1, 0, 0]])]
        em_pivot_data = {"frames": frames}
        
        with pytest.raises(ValueError, match="EM pivot data must contain at least 2 frames"):
            em_pivot_calibration(em_pivot_data)
    
    def test_em_pivot_calibration_list_input(self):
        """Test EM pivot calibration with list input"""
        frames = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]])
        ]
        
        result = em_pivot_calibration(frames)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result


class TestOptPivotCalibration:
    """Test cases for optical pivot calibration algorithm"""
    
    def test_opt_pivot_calibration_dict_input(self):
        """Test optical pivot calibration with dictionary input"""
        frames = [
            {"h_points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])},
            {"h_points": np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]])}
        ]
        
        opt_pivot_data = {"frames": frames}
        
        result = opt_pivot_calibration(opt_pivot_data)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        
        assert len(result['tip_position']) == 3
        assert len(result['pivot_point']) == 3
        assert isinstance(result['residual_error'], (int, float))
    
    def test_opt_pivot_calibration_list_input(self):
        """Test optical pivot calibration with list input"""
        frames = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]])
        ]
        
        result = opt_pivot_calibration(frames)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
    
    def test_opt_pivot_calibration_insufficient_frames(self):
        """Test optical pivot calibration with insufficient frames"""
        frames = [np.array([[0, 0, 0], [1, 0, 0]])]
        
        with pytest.raises(ValueError, match="Need at least two frames for pivot calibration"):
            opt_pivot_calibration(frames)


class TestPivotCalibrationAlgorithmConsistency:
    """Test consistency between EM and optical pivot calibration algorithms"""
    
    def test_em_vs_opt_pivot_consistency(self):
        """Test that EM and optical pivot calibration produce consistent results for same data"""
        # Create identical test data
        frames = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
            np.array([[0.2, 0.2, 0.2], [1.2, 0.2, 0.2], [0.2, 1.2, 0.2]])
        ]
        
        em_result = em_pivot_calibration(frames)
        opt_result = opt_pivot_calibration(frames)
        
        # Results should be similar (within numerical precision)
        assert_array_almost_equal(em_result['tip_position'], opt_result['tip_position'], decimal=5)
        assert_array_almost_equal(em_result['pivot_point'], opt_result['pivot_point'], decimal=5)
    
    def test_pivot_calibration_residual_error(self):
        """Test that residual error is reasonable"""
        # Create test data with known transformation
        frames = [
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
            np.array([[0.2, 0.2, 0.2], [1.2, 0.2, 0.2], [0.2, 1.2, 0.2]])
        ]
        
        result = em_pivot_calibration(frames)
        
        # Residual error should be non-negative and reasonable
        assert result['residual_error'] >= 0
        assert result['residual_error'] < 1000  # Should not be extremely large


if __name__ == "__main__":
    pytest.main([__file__])