"""
Unit tests for Distortion Correction Module

Tests the distortion correction algorithms using synthetic data with known distortions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from distortion_correction import (
    fit_distortion_correction,
    apply_distortion_correction,
    correct_frame_markers
)


class TestDistortionCorrection:
    """Test cases for distortion correction fitting and application"""

    def test_fit_zero_distortion(self):
        """Test fitting with zero distortion (identity correction)"""
        # Generate points with no distortion
        np.random.seed(42)
        points = np.random.rand(50, 3) * 100 + 100  # Random points in [100, 200]^3

        # Distorted = expected (no distortion)
        model = fit_distortion_correction(points, points, degree=3)

        # Apply correction
        corrected = apply_distortion_correction(points, model)

        # Should recover original points
        error = np.linalg.norm(corrected - points, axis=1)
        assert np.mean(error) < 0.1  # Very small error

    def test_fit_linear_distortion(self):
        """Test fitting with simple linear distortion"""
        # Generate points
        np.random.seed(42)
        expected = np.random.rand(30, 3) * 100 + 100

        # Add linear distortion: distorted = expected + offset
        offset = np.array([5.0, -3.0, 2.0])
        distorted = expected + offset

        # Fit correction
        model = fit_distortion_correction(distorted, expected, degree=3)

        # Apply correction
        corrected = apply_distortion_correction(distorted, model)

        # Should recover expected points
        error = np.linalg.norm(corrected - expected, axis=1)
        assert np.mean(error) < 1.0  # Should correct linear distortion well

    def test_fit_quadratic_distortion(self):
        """Test fitting with quadratic distortion"""
        # Generate grid of points
        x = np.linspace(100, 200, 5)
        y = np.linspace(200, 300, 5)
        z = np.linspace(300, 400, 5)
        xx, yy, zz = np.meshgrid(x, y, z)
        expected = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Add quadratic distortion
        distortion_x = 0.001 * (expected[:, 0] - 150) ** 2
        distortion_y = 0.001 * (expected[:, 1] - 250) ** 2
        distortion_z = 0.001 * (expected[:, 2] - 350) ** 2
        distorted = expected + np.column_stack([distortion_x, distortion_y, distortion_z])

        # Fit correction with sufficient degree
        model = fit_distortion_correction(distorted, expected, degree=5)

        # Apply correction
        corrected = apply_distortion_correction(distorted, model)

        # Should significantly reduce distortion
        error_before = np.linalg.norm(distorted - expected, axis=1)
        error_after = np.linalg.norm(corrected - expected, axis=1)

        assert np.mean(error_after) < np.mean(error_before) * 0.1  # At least 90% improvement

    def test_correction_single_point(self):
        """Test that correction works for single points"""
        # Generate training data
        np.random.seed(42)
        expected = np.random.rand(30, 3) * 100 + 100
        distorted = expected + np.array([2.0, -1.0, 3.0])

        # Fit model
        model = fit_distortion_correction(distorted, expected, degree=3)

        # Test single point correction
        test_point = np.array([150.0, 250.0, 350.0])
        corrected = apply_distortion_correction(test_point, model)

        # Should return array of shape (3,)
        assert corrected.shape == (3,)

    def test_correct_frame_markers(self):
        """Test correction of frame markers"""
        # Generate training data
        np.random.seed(42)
        expected = np.random.rand(30, 3) * 100 + 100
        distorted = expected + 5.0  # Uniform offset

        # Fit model
        model = fit_distortion_correction(distorted, expected, degree=3)

        # Test frame correction
        frame = np.array([[150, 250, 350], [160, 260, 360], [170, 270, 370]])
        corrected = correct_frame_markers(frame, model)

        # Should return same shape
        assert corrected.shape == frame.shape
        assert corrected.shape == (3, 3)


class TestDistortionCorrectionProperties:
    """Test mathematical properties of distortion correction"""

    def test_correction_consistency(self):
        """Test that applying correction twice gives same result"""
        np.random.seed(42)
        expected = np.random.rand(20, 3) * 100
        distorted = expected + 3.0

        model = fit_distortion_correction(distorted, expected, degree=3)

        # Apply correction
        corrected1 = apply_distortion_correction(distorted, model)
        corrected2 = apply_distortion_correction(distorted, model)

        # Should be identical
        assert_array_almost_equal(corrected1, corrected2, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
