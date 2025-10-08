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
    
    def test_solve_for_pivot_ground_truth_validation(self):
        """Test solve_for_pivot with known ground truth - validates least squares accuracy"""
        # Create synthetic data with KNOWN tip and pivot positions
        true_tip = np.array([1.0, 2.0, 3.0])
        true_pivot = np.array([4.0, 5.0, 6.0])
        
        # Generate multiple frames with diverse rotations around different axes
        R_list = []
        p_list = []
        
        # Rotation around X-axis
        angles_x = [0, np.pi/4, np.pi/2]
        for angle in angles_x:
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
            p_i = true_pivot - R @ true_tip
            R_list.append(R)
            p_list.append(p_i)
        
        # Rotation around Y-axis
        angles_y = [np.pi/6, np.pi/3]
        for angle in angles_y:
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
            p_i = true_pivot - R @ true_tip
            R_list.append(R)
            p_list.append(p_i)
        
        # Rotation around Z-axis
        angles_z = [np.pi/8, 3*np.pi/4]
        for angle in angles_z:
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
            p_i = true_pivot - R @ true_tip
            R_list.append(R)
            p_list.append(p_i)
        
        # Solve using least squares
        p_tip_result, p_pivot_result, residual_error = solve_for_pivot(R_list, p_list)
        
        # CRITICAL: Validate that least squares recovered the ground truth
        assert_array_almost_equal(p_tip_result, true_tip, decimal=5)
        assert_array_almost_equal(p_pivot_result, true_pivot, decimal=5)
        
        # Residual should be very small for perfect synthetic data
        assert residual_error < 1e-10
    
    def test_solve_for_pivot_with_noise(self):
        """Test solve_for_pivot robustness with synthetic noise"""
        # Known ground truth
        true_tip = np.array([1.0, 2.0, 3.0])
        true_pivot = np.array([4.0, 5.0, 6.0])
        
        # Generate frames with noise
        np.random.seed(42)  # For reproducible tests
        angles = np.linspace(0, 2*np.pi, 10)
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
        assert np.linalg.norm(p_tip_result - true_tip) < 0.1  # Within noise level
        assert np.linalg.norm(p_pivot_result - true_pivot) < 0.1
        assert residual_error > 0  # Should have some residual due to noise
    
    def test_solve_for_pivot_convergence(self):
        """Test that solve_for_pivot converges with increasing number of frames"""
        true_tip = np.array([0.0, 0.0, 0.0])
        true_pivot = np.array([1.0, 0.0, 0.0])
        
        # Test with different numbers of frames
        frame_counts = [3, 5, 10, 20]
        errors_tip = []
        errors_pivot = []
        
        for n_frames in frame_counts:
            angles = np.linspace(0, 2*np.pi, n_frames)
            R_list = []
            p_list = []
            
            for angle in angles:
                R = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]])
                # Correct equation: p_i = p_pivot - R_i * p_tip
                p_i = true_pivot - R @ true_tip
                R_list.append(R)
                p_list.append(p_i)
            
            p_tip_result, p_pivot_result, _ = solve_for_pivot(R_list, p_list)
            
            errors_tip.append(np.linalg.norm(p_tip_result - true_tip))
            errors_pivot.append(np.linalg.norm(p_pivot_result - true_pivot))
        
        # Error should generally decrease with more frames (better conditioning)
        # At minimum, should be accurate with sufficient frames
        assert errors_tip[-1] < 1e-10  # Very accurate with 20 frames
        assert errors_pivot[-1] < 1e-10
    
    def test_solve_for_pivot_deterministic_ground_truth(self):
        """Deterministic ground-truth checks: verify recovered tip/pivot are close to known truth on synthetic data"""
        # Test multiple known configurations
        test_cases = [
            {"tip": np.array([0.0, 0.0, 0.0]), "pivot": np.array([1.0, 0.0, 0.0])},
            {"tip": np.array([1.0, 2.0, 3.0]), "pivot": np.array([4.0, 5.0, 6.0])},
            {"tip": np.array([-2.5, 1.0, -3.0]), "pivot": np.array([0.5, 4.0, 2.0])},
            {"tip": np.array([10.0, -5.0, 8.0]), "pivot": np.array([15.0, 0.0, 12.0])},
        ]
        
        for i, case in enumerate(test_cases):
            true_tip = case["tip"]
            true_pivot = case["pivot"]
            
            # Generate diverse rotations
            R_list = []
            p_list = []
            
            # Rotations around different axes
            angles = np.linspace(0, 2*np.pi, 8)
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
            
            # CRITICAL: Validate deterministic ground truth recovery
            tip_error = np.linalg.norm(p_tip_result - true_tip)
            pivot_error = np.linalg.norm(p_pivot_result - true_pivot)
            
            assert tip_error < 1e-10, f"Case {i}: Tip error {tip_error} too large"
            assert pivot_error < 1e-10, f"Case {i}: Pivot error {pivot_error} too large"
            assert residual_error < 1e-10, f"Case {i}: Residual error {residual_error} too large"
    
    def test_solve_for_pivot_randomized_roundtrip(self):
        """Test randomized roundtrip with known t_tip and p_pivot"""
        np.random.seed(42)
        
        for trial in range(15):
            # Generate random known truth
            true_tip = np.random.randn(3) * 5
            true_pivot = np.random.randn(3) * 5
            
            # Generate random rotations
            n_frames = np.random.randint(5, 15)
            R_list = []
            p_list = []
            
            for _ in range(n_frames):
                # Random SO(3) rotation
                random_rotation = np.random.randn(3, 3)
                U, _, Vt = np.linalg.svd(random_rotation)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    R = U @ np.diag([1, 1, -1]) @ Vt
                
                # Correct equation: p_i = p_pivot - R_i * p_tip
                p_i = true_pivot - R @ true_tip
                
                R_list.append(R)
                p_list.append(p_i)
            
            # Solve using least squares
            p_tip_result, p_pivot_result, residual_error = solve_for_pivot(R_list, p_list)
            
            # Verify ground truth recovery
            tip_error = np.linalg.norm(p_tip_result - true_tip)
            pivot_error = np.linalg.norm(p_pivot_result - true_pivot)
            
            assert tip_error < 1e-10, f"Trial {trial}: Tip error {tip_error} too large"
            assert pivot_error < 1e-10, f"Trial {trial}: Pivot error {pivot_error} too large"
            assert residual_error < 1e-10, f"Trial {trial}: Residual error {residual_error} too large"


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
    """Test cases for optical pivot calibration algorithm with integrated optical→EM transformation"""
    
    def test_opt_pivot_calibration_dict_input(self):
        """Test optical pivot calibration with dictionary input"""
        frames = [
            {"h_points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "d_points": np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]])},
            {"h_points": np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
             "d_points": np.array([[10.1, 10.1, 10.1], [11.1, 10.1, 10.1], [10.1, 11.1, 10.1]])}
        ]
        
        opt_pivot_data = {"frames": frames}
        
        result = opt_pivot_calibration(opt_pivot_data)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        
        assert len(result['tip_position']) == 3
        assert len(result['pivot_point']) == 3
        assert isinstance(result['residual_error'], (int, float))
    
    def test_opt_pivot_calibration_with_cal_body(self):
        """Test optical pivot calibration with calibration body data"""
        frames = [
            {"h_points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "d_points": np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]])},
            {"h_points": np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
             "d_points": np.array([[10.1, 10.1, 10.1], [11.1, 10.1, 10.1], [10.1, 11.1, 10.1]])}
        ]
        
        opt_pivot_data = {"frames": frames}
        cal_body_data = {"d_points": np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]])}
        
        result = opt_pivot_calibration(opt_pivot_data, cal_body_data)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        
        assert len(result['tip_position']) == 3
        assert len(result['pivot_point']) == 3
        assert isinstance(result['residual_error'], (int, float))
    
    def test_opt_pivot_calibration_insufficient_frames(self):
        """Test optical pivot calibration with insufficient frames"""
        frames = [{"h_points": np.array([[0, 0, 0], [1, 0, 0]]),
                  "d_points": np.array([[10, 10, 10], [11, 10, 10]])}]
        opt_pivot_data = {"frames": frames}
        
        with pytest.raises(AssertionError):
            opt_pivot_calibration(opt_pivot_data)
    
    def test_opt_pivot_calibration_integrated_transformation(self):
        """Test the integrated optical→EM transformation approach"""
        # Create synthetic data with known transformations
        np.random.seed(42)
        
        # EM-base geometry (known ground truth)
        d_em = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        
        # Probe local shape (zero-mean)
        h_local = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], dtype=float)
        
        # Generate multiple frames with known transformations
        frames = []
        true_tip = np.array([0.1, 0.2, 0.3])
        true_pivot = np.array([1.0, 2.0, 3.0])
        
        for i in range(5):
            # Create optical measurements with known transformation
            angle = i * np.pi / 4
            R_optical = np.array([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0],
                                  [0, 0, 1]])
            t_optical = np.array([i * 0.1, i * 0.2, i * 0.3])
            
            # Transform EM-base and probe points
            d_optical = (R_optical @ d_em.T).T + t_optical
            h_optical = (R_optical @ h_local.T).T + t_optical
            
            frames.append({
                "d_points": d_optical,
                "h_points": h_optical
            })
        
        opt_pivot_data = {"frames": frames}
        cal_body_data = {"d_points": d_em}
        
        result = opt_pivot_calibration(opt_pivot_data, cal_body_data)
        
        # Verify results are reasonable
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        
        # Residual should be small for synthetic data
        assert result['residual_error'] < 1.0
        
        # Results should be finite
        assert np.all(np.isfinite(result['tip_position']))
        assert np.all(np.isfinite(result['pivot_point']))
    
    def test_opt_pivot_calibration_coordinate_system_consistency(self):
        """Test that optical pivot calibration maintains coordinate system consistency"""
        # Create data where EM-base markers are stationary (common case)
        d_em = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        
        frames = []
        for i in range(3):
            # EM-base markers stay the same (stationary)
            d_optical = d_em.copy()
            
            # Probe markers move (simulating probe motion)
            h_optical = np.array([[i * 0.1, i * 0.2, i * 0.3],
                                 [i * 0.1 + 0.5, i * 0.2, i * 0.3],
                                 [i * 0.1, i * 0.2 + 0.5, i * 0.3]], dtype=float)
            
            frames.append({
                "d_points": d_optical,
                "h_points": h_optical
            })
        
        opt_pivot_data = {"frames": frames}
        cal_body_data = {"d_points": d_em}
        
        result = opt_pivot_calibration(opt_pivot_data, cal_body_data)
        
        # Should produce reasonable results even with stationary EM-base
        assert result['residual_error'] >= 0
        assert np.all(np.isfinite(result['tip_position']))
        assert np.all(np.isfinite(result['pivot_point']))
    
    def test_opt_pivot_calibration_fallback_mode(self):
        """Test optical pivot calibration fallback mode (no cal_body_data)"""
        frames = [
            {"h_points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
             "d_points": np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]])},
            {"h_points": np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
             "d_points": np.array([[10.1, 10.1, 10.1], [11.1, 10.1, 10.1], [10.1, 11.1, 10.1]])}
        ]
        
        opt_pivot_data = {"frames": frames}
        
        # Test without cal_body_data (should use first frame as fallback)
        result = opt_pivot_calibration(opt_pivot_data)
        
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result


class TestPivotCalibrationAlgorithmConsistency:
    """Test consistency between EM and optical pivot calibration algorithms"""
    
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
    
    def test_optical_pivot_calibration_accuracy_with_real_data_structure(self):
        """Test optical pivot calibration with realistic data structure"""
        # Create realistic optical pivot data structure
        frames = []
        for i in range(5):
            # Simulate realistic marker positions
            d_points = np.array([
                [100 + i*0.1, 200 + i*0.2, 300 + i*0.3],
                [101 + i*0.1, 200 + i*0.2, 300 + i*0.3],
                [100 + i*0.1, 201 + i*0.2, 300 + i*0.3]
            ], dtype=float)
            
            h_points = np.array([
                [400 + i*0.5, 500 + i*0.6, 600 + i*0.7],
                [401 + i*0.5, 500 + i*0.6, 600 + i*0.7],
                [400 + i*0.5, 501 + i*0.6, 600 + i*0.7]
            ], dtype=float)
            
            frames.append({
                "d_points": d_points,
                "h_points": h_points
            })
        
        opt_pivot_data = {"frames": frames}
        
        # Test with calibration body data
        cal_body_data = {"d_points": frames[0]["d_points"]}
        
        result = opt_pivot_calibration(opt_pivot_data, cal_body_data)
        
        # Verify results are reasonable
        assert 'tip_position' in result
        assert 'pivot_point' in result
        assert 'residual_error' in result
        
        # Results should be finite and in reasonable ranges
        assert np.all(np.isfinite(result['tip_position']))
        assert np.all(np.isfinite(result['pivot_point']))
        assert result['residual_error'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])