"""
Unit tests for FrameTransform class

Tests core coordinate frame transformation algorithms:
- Point Set Registration algorithm
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
    
    def test_point_set_registration_with_weights(self):
        """Test registration with weighted points"""
        points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        translation = np.array([1, 1, 1])
        points2 = points1 + translation
        
        # Give higher weight to first two points
        weights = np.array([2, 2, 1, 1], dtype=float)
        
        transform = FrameTransform.Point_set_registration(points1, points2, weights)
        
        # Should still recover the translation
        assert_array_almost_equal(transform.rotation_matrix, np.eye(3), decimal=10)
        assert_array_almost_equal(transform.translation_vector, translation, decimal=10)
    
    def test_point_set_registration_reflection_case(self):
        """Test registration handles reflection case correctly"""
        # Create points that would result in reflection
        points1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        points2 = np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]], dtype=float)  # Reflected
        
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Check that determinant is positive (no reflection)
        assert np.linalg.det(transform.rotation_matrix) > 0
        
        # Test that transformation works
        transformed_points = transform.transform_points(points1)
        assert_array_almost_equal(transformed_points, points2, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__])
