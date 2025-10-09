"""
Unit tests for FrameTransform class

Tests core coordinate frame transformation algorithms using synthetic data:
- Point Set Registration algorithm (SVD-based least squares)
- Basic transformation operations (transform, inverse, compose)
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from frame_transform import FrameTransform


class TestFrameTransformCore:
    """Test cases for core FrameTransform operations"""
    
    def test_transform_points(self):
        """Test transformation of multiple points"""
        # Identity transformation
        transform = FrameTransform()
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = transform.transform_points(points)
        assert_array_equal(result, points)
        
        # Translation only
        transform = FrameTransform(translation_vector=np.array([1, 1, 1]))
        points = np.array([[0, 0, 0], [1, 1, 1]])
        result = transform.transform_points(points)
        expected = np.array([[1, 1, 1], [2, 2, 2]])
        assert_array_equal(result, expected)
        
        # Rotation and translation
        angle = np.pi / 2
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([1, 0, 0])
        transform = FrameTransform(R, t)
        points = np.array([[1, 0, 0], [0, 1, 0]])
        result = transform.transform_points(points)
        expected = np.array([[1, 1, 0], [0, 0, 0]])
        assert_array_almost_equal(result, expected, decimal=10)
    
    def test_inverse_general(self):
        """Test inverse of general transformation"""
        angle = np.pi / 3
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([1, 2, 3])
        transform = FrameTransform(R, t)
        inverse = transform.inverse()
        
        # Test that inverse * original = identity
        composed = transform.compose(inverse)
        assert_array_almost_equal(composed.rotation_matrix, np.eye(3), decimal=10)
        assert_array_almost_equal(composed.translation_vector, np.zeros(3), decimal=10)
    
    def test_compose_general(self):
        """Test composition of general transformations"""
        angle = np.pi / 3
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t1 = np.array([1, 2, 3])
        t2 = np.array([4, 5, 6])
        
        transform1 = FrameTransform(R, t1)
        transform2 = FrameTransform(R, t2)
        result = transform1.compose(transform2)
        
        expected_R = R @ R
        expected_t = R @ t2 + t1
        assert_array_almost_equal(result.rotation_matrix, expected_R, decimal=10)
        assert_array_almost_equal(result.translation_vector, expected_t, decimal=10)


class TestPointSetRegistration:
    """Test cases for Point Set Registration algorithm"""
    
    def test_point_set_registration_identity(self):
        """Test registration with identical point sets"""
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        transform = FrameTransform.Point_set_registration(points, points)
        
        # Should result in identity transformation
        assert_array_almost_equal(transform.rotation_matrix, np.eye(3), decimal=10)
        assert_array_almost_equal(transform.translation_vector, np.zeros(3), decimal=10)
    
    def test_point_set_registration_translation(self):
        """Test registration with translated point sets"""
        points1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        translation = np.array([2, 3, 4])
        points2 = points1 + translation
        
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Should recover the translation
        assert_array_almost_equal(transform.rotation_matrix, np.eye(3), decimal=10)
        assert_array_almost_equal(transform.translation_vector, translation, decimal=10)
    
    def test_point_set_registration_rotation(self):
        """Test registration with rotated point sets"""
        # Create points in a square
        points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        
        # Rotate 90 degrees around Z-axis
        angle = np.pi / 2
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        points2 = (R @ points1.T).T
        
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Should recover the rotation (within numerical precision)
        assert_array_almost_equal(transform.rotation_matrix, R, decimal=10)
        assert_array_almost_equal(transform.translation_vector, np.zeros(3), decimal=10)
    
    def test_point_set_registration_general(self):
        """Test registration with general transformation"""
        points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        
        # Apply rotation and translation
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([2, 3, 4])
        points2 = (R @ points1.T).T + t
        
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Test that the transformation works
        transformed_points = transform.transform_points(points1)
        assert_array_almost_equal(transformed_points, points2, decimal=10)
    
    def test_point_set_registration_least_squares_accuracy(self):
        """Test Point Set Registration least squares accuracy with complex synthetic data"""
        # Create a complex 3D point set (tetrahedron)
        points1 = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [0.5, np.sqrt(3)/2, 0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
        ], dtype=float)
        
        # Apply complex transformation: rotation + translation
        # Rotation around axis [1, 1, 1] by 60 degrees
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 3
        
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_true = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        t_true = np.array([2.5, -1.0, 3.0])
        
        # Apply transformation to get points2
        points2 = (R_true @ points1.T).T + t_true
        
        # Use Point Set Registration (SVD-based least squares)
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Validate that SVD least squares recovered the ground truth
        assert_array_almost_equal(transform.rotation_matrix, R_true, decimal=8)
        assert_array_almost_equal(transform.translation_vector, t_true, decimal=8)
        
        # Verify the transformation works end-to-end
        transformed_points = transform.transform_points(points1)
        assert_array_almost_equal(transformed_points, points2, decimal=10)
    
    def test_point_set_registration_with_noise(self):
        """Test Point Set Registration robustness with synthetic noise"""
        # Known ground truth transformation
        R_true = np.array([[0.8, -0.6, 0],
                          [0.6, 0.8, 0],
                          [0, 0, 1]], dtype=float)
        t_true = np.array([1.0, 2.0, 3.0])
        
        # Create point set
        points1 = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=float)
        
        # Apply transformation
        points2_clean = (R_true @ points1.T).T + t_true
        
        # Add noise
        np.random.seed(42)
        noise_level = 0.01  # 1% noise
        noise = noise_level * np.random.randn(*points2_clean.shape)
        points2_noisy = points2_clean + noise
        
        # Test registration with noisy data
        transform = FrameTransform.Point_set_registration(points1, points2_noisy)
        
        # Should recover transformation within noise tolerance
        rotation_error = np.linalg.norm(transform.rotation_matrix - R_true, 'fro')
        translation_error = np.linalg.norm(transform.translation_vector - t_true)
        
        assert rotation_error < 0.1  # Within noise level
        assert translation_error < 0.1
        
        # Verify transformation quality
        transformed_points = transform.transform_points(points1)
        reconstruction_error = np.linalg.norm(transformed_points - points2_noisy, axis=1).mean()
        assert reconstruction_error < noise_level * 2  # Should be close to noise level
    
    def test_point_set_registration_orthonormality(self):
        """Test that registration results satisfy orthonormality: R.T @ R ≈ I and det(R) ≈ +1"""
        np.random.seed(42)
        
        for _ in range(5):
            # Generate random point sets
            n_points = np.random.randint(4, 10)
            points1 = np.random.randn(n_points, 3)
            
            # Apply random transformation
            random_rotation = np.random.randn(3, 3)
            U, _, Vt = np.linalg.svd(random_rotation)
            R_true = U @ Vt
            if np.linalg.det(R_true) < 0:
                R_true = U @ np.diag([1, 1, -1]) @ Vt
            
            t_true = np.random.randn(3)
            points2 = (R_true @ points1.T).T + t_true
            
            # Perform registration
            transform = FrameTransform.Point_set_registration(points1, points2)
            R_result = transform.rotation_matrix
            
            # Test orthonormality: R.T @ R ≈ I
            orthogonality_check = R_result.T @ R_result
            assert_array_almost_equal(orthogonality_check, np.eye(3), decimal=10)
            
            # Test determinant: det(R) ≈ +1
            det_R = np.linalg.det(R_result)
            assert abs(det_R - 1.0) < 1e-10, f"Determinant should be +1, got {det_R}"
            
            # Test that R_result is a proper rotation matrix
            assert det_R > 0, "Determinant should be positive for proper rotation"


if __name__ == "__main__":
    pytest.main([__file__])
