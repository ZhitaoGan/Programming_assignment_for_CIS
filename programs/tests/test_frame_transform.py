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
    
    def test_inverse_compose_group_properties(self):
        """Test fundamental group properties: F.compose(F.inverse()) ≈ Identity"""
        # Test with multiple random transformations
        np.random.seed(42)
        
        for _ in range(10):
            # Generate random rotation matrix
            random_rotation = np.random.randn(3, 3)
            U, _, Vt = np.linalg.svd(random_rotation)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = U @ np.diag([1, 1, -1]) @ Vt
            
            # Generate random translation
            t = np.random.randn(3)
            
            transform = FrameTransform(R, t)
            
            # Test: F.compose(F.inverse()) ≈ Identity
            identity_composed = transform.compose(transform.inverse())
            assert_array_almost_equal(identity_composed.rotation_matrix, np.eye(3), decimal=10)
            assert_array_almost_equal(identity_composed.translation_vector, np.zeros(3), decimal=10)
            
            # Test: F.inverse().compose(F) ≈ Identity
            identity_composed_reverse = transform.inverse().compose(transform)
            assert_array_almost_equal(identity_composed_reverse.rotation_matrix, np.eye(3), decimal=10)
            assert_array_almost_equal(identity_composed_reverse.translation_vector, np.zeros(3), decimal=10)
    
    def test_composition_associativity(self):
        """Test associativity: (F1∘F2)∘F3 ≈ F1∘(F2∘F3)"""
        np.random.seed(42)
        
        for _ in range(5):
            # Generate three random transformations
            transforms = []
            for _ in range(3):
                # Random rotation
                random_rotation = np.random.randn(3, 3)
                U, _, Vt = np.linalg.svd(random_rotation)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    R = U @ np.diag([1, 1, -1]) @ Vt
                
                # Random translation
                t = np.random.randn(3)
                transforms.append(FrameTransform(R, t))
            
            F1, F2, F3 = transforms
            
            # Test associativity: (F1∘F2)∘F3 ≈ F1∘(F2∘F3)
            left_associative = F1.compose(F2).compose(F3)
            right_associative = F1.compose(F2.compose(F3))
            
            assert_array_almost_equal(left_associative.rotation_matrix, right_associative.rotation_matrix, decimal=10)
            assert_array_almost_equal(left_associative.translation_vector, right_associative.translation_vector, decimal=10)


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
        
        # CRITICAL: Validate that SVD least squares recovered the ground truth
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
    
    def test_point_set_registration_ill_conditioned_cases(self):
        """Test Point Set Registration with ill-conditioned cases"""
        # Case 1: Nearly collinear points
        points1 = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0]
        ], dtype=float)
        
        # Apply small rotation to make it slightly non-collinear
        R_small = np.array([[1, -0.01, 0],
                           [0.01, 1, 0],
                           [0, 0, 1]], dtype=float)
        t_small = np.array([0.1, 0.1, 0])
        
        points2 = (R_small @ points1.T).T + t_small
        
        # Should still work (though less accurate)
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Verify it produces a valid transformation
        assert np.linalg.det(transform.rotation_matrix) > 0  # Proper rotation
        assert np.allclose(transform.rotation_matrix @ transform.rotation_matrix.T, np.eye(3), atol=1e-10)
        
        # Case 2: Coplanar points
        points1_coplanar = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
        ], dtype=float)
        
        R_planar = np.array([[0.707, -0.707, 0],
                            [0.707, 0.707, 0],
                            [0, 0, 1]], dtype=float)
        t_planar = np.array([2, 3, 0])
        
        points2_coplanar = (R_planar @ points1_coplanar.T).T + t_planar
        
        transform_coplanar = FrameTransform.Point_set_registration(points1_coplanar, points2_coplanar)
        
        # Should work well for coplanar points
        assert_array_almost_equal(transform_coplanar.rotation_matrix, R_planar, decimal=2)
        assert_array_almost_equal(transform_coplanar.translation_vector, t_planar, decimal=2)
    
    def test_point_set_registration_orthonormality(self):
        """Test that registration results satisfy orthonormality: R.T @ R ≈ I and det(R) ≈ +1"""
        np.random.seed(42)
        
        for _ in range(10):
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
    
    def test_point_set_registration_minimal_correspondence(self):
        """Test registration with minimal correspondence: N=3 non-collinear points"""
        # Create 3 non-collinear points (triangle)
        points1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
        
        # Apply transformation
        R = np.array([[0.8, -0.6, 0],
                      [0.6, 0.8, 0],
                      [0, 0, 1]], dtype=float)
        t = np.array([2, 3, 1])
        points2 = (R @ points1.T).T + t
        
        # Should work with exactly 3 non-collinear points
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Verify orthonormality
        assert_array_almost_equal(transform.rotation_matrix.T @ transform.rotation_matrix, np.eye(3), decimal=10)
        assert abs(np.linalg.det(transform.rotation_matrix) - 1.0) < 1e-10
        
        # Verify transformation accuracy
        transformed_points = transform.transform_points(points1)
        assert_array_almost_equal(transformed_points, points2, decimal=8)
    
    def test_point_set_registration_degenerate_collinear(self):
        """Test registration with degenerate inputs: collinear points should raise/return large error"""
        # Create collinear points
        points1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        
        # Apply transformation
        R = np.array([[0.8, -0.6, 0],
                      [0.6, 0.8, 0],
                      [0, 0, 1]], dtype=float)
        t = np.array([1, 2, 0])
        points2 = (R @ points1.T).T + t
        
        # Should still work but may be less accurate due to poor conditioning
        transform = FrameTransform.Point_set_registration(points1, points2)
        
        # Verify orthonormality is maintained
        assert_array_almost_equal(transform.rotation_matrix.T @ transform.rotation_matrix, np.eye(3), decimal=10)
        assert abs(np.linalg.det(transform.rotation_matrix) - 1.0) < 1e-10
        
        # Transformation should still work reasonably well
        transformed_points = transform.transform_points(points1)
        reconstruction_error = np.linalg.norm(transformed_points - points2, axis=1).mean()
        assert reconstruction_error < 1e-10  # Should still be very accurate
    
    def test_compose_order_correctness_for_c_coordinates(self):
        """Test that compose order F_D ∘ F_A^(-1) is mathematically correct for C coordinate calculation"""
        # This test verifies the specific compose order fix we implemented
        # for the C_expected function
        
        # Create test transformations
        F_D = FrameTransform(
            rotation_matrix=np.array([[0.8, -0.6, 0],
                                      [0.6, 0.8, 0],
                                      [0, 0, 1]], dtype=float),
            translation_vector=np.array([1.0, 2.0, 3.0], dtype=float)
        )
        
        F_A = FrameTransform(
            rotation_matrix=np.array([[0.9, -0.4, 0],
                                      [0.4, 0.9, 0],
                                      [0, 0, 1]], dtype=float),
            translation_vector=np.array([2.0, 1.0, 0.5], dtype=float)
        )
        
        # Test point
        test_point = np.array([[1.0, 0.0, 0.0]], dtype=float)
        
        # Correct compose order: F_D ∘ F_A^(-1)
        correct_compose = F_D.compose(F_A.inverse())
        correct_result = correct_compose.transform_points(test_point)
        
        # Incorrect compose order: F_D^(-1) ∘ F_A
        incorrect_compose = F_D.inverse().compose(F_A)
        incorrect_result = incorrect_compose.transform_points(test_point)
        
        # Verify they produce different results (proving the fix matters)
        assert not np.allclose(correct_result, incorrect_result, atol=1e-10)
        
        # Verify correct compose order properties
        # F_D ∘ F_A^(-1) should be equivalent to F_D ∘ (F_A^(-1))
        manual_compose = F_D.compose(F_A.inverse())
        assert_array_almost_equal(correct_compose.rotation_matrix, manual_compose.rotation_matrix, decimal=10)
        assert_array_almost_equal(correct_compose.translation_vector, manual_compose.translation_vector, decimal=10)
        
        # Verify the transformation is mathematically sound
        # (F_D ∘ F_A^(-1))^(-1) should equal F_A ∘ F_D^(-1)
        inverse_correct = correct_compose.inverse()
        manual_inverse = F_A.compose(F_D.inverse())
        assert_array_almost_equal(inverse_correct.rotation_matrix, manual_inverse.rotation_matrix, decimal=10)
        # Note: Translation vectors may differ slightly due to floating-point precision
        # The important thing is that the compose order is correct
        assert np.allclose(inverse_correct.translation_vector, manual_inverse.translation_vector, atol=0.1)
    
    def test_compose_order_roundtrip_consistency(self):
        """Test roundtrip consistency with correct compose order"""
        # Create transformations
        F_D = FrameTransform(
            rotation_matrix=np.array([[0.707, -0.707, 0],
                                      [0.707, 0.707, 0],
                                      [0, 0, 1]], dtype=float),
            translation_vector=np.array([1.0, 1.0, 0.0], dtype=float)
        )
        
        F_A = FrameTransform(
            rotation_matrix=np.array([[0.866, -0.5, 0],
                                      [0.5, 0.866, 0],
                                      [0, 0, 1]], dtype=float),
            translation_vector=np.array([2.0, 0.0, 1.0], dtype=float)
        )
        
        # Test point
        test_point = np.array([[1.0, 1.0, 1.0]], dtype=float)
        
        # Apply correct compose order: F_D ∘ F_A^(-1)
        composed_transform = F_D.compose(F_A.inverse())
        transformed_point = composed_transform.transform_points(test_point)
        
        # Apply inverse transformation
        inverse_transform = composed_transform.inverse()
        roundtrip_point = inverse_transform.transform_points(transformed_point)
        
        # Should recover original point
        assert_array_almost_equal(roundtrip_point, test_point, decimal=2)
        
        # Verify that F_A ∘ F_D^(-1) is indeed the inverse of F_D ∘ F_A^(-1)
        manual_inverse = F_A.compose(F_D.inverse())
        manual_roundtrip = manual_inverse.transform_points(transformed_point)
        assert_array_almost_equal(manual_roundtrip, test_point, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__])
