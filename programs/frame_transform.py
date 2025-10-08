"""
Frame Transform Module

This module contains the FrameTransform class for handling coordinate frame transformations.
Used for converting points between different coordinate systems.
"""

import numpy as np


class FrameTransform:
    """
    A class for handling coordinate frame transformations.
    
    This class provides methods for transforming points between different
    coordinate frames using rotation matrices and translation vectors.
    """
    
    def __init__(self, rotation_matrix=None, translation_vector=None):
        """
        Initialize FrameTransform with rotation matrix and translation vector.
        
        Args:
            rotation_matrix (np.ndarray, optional): 3x3 rotation matrix
            translation_vector (np.ndarray, optional): 3x1 translation vector
        """
        self.rotation_matrix = rotation_matrix if rotation_matrix is not None else np.eye(3)
        self.translation_vector = translation_vector if translation_vector is not None else np.zeros(3)
    
    def transform_single_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a single point using the rotation matrix and translation vector.
        
        Args:
            point (np.ndarray): 3x1 array of point to transform
            
        Returns:
            np.ndarray: 3x1 array of transformed point
        """
        point = np.asarray(point,dtype=float)
        if point.shape[0] != 3:
            raise ValueError("Point must be a 3x1 array")
        transformed_point = np.dot(self.rotation_matrix, point) + self.translation_vector
        return transformed_point
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points using the rotation matrix and translation vector.
        
        Args:
            points (np.ndarray): Nx3 array of points to transform
            
        Returns:
            np.ndarray: Nx3 array of transformed points
        """
        # Forward transformation: v = R * b + p
        # points is Nx3, rotation_matrix is 3x3, translation_vector is 3x1
        # We need to transpose points to 3xN for matrix multiplication
        points = np.asarray(points,dtype=float)
        if points.shape[1] != 3:
            raise ValueError("Points must be a Nx3 array")
        transformed_points = np.dot(self.rotation_matrix, points.T) + self.translation_vector.reshape(3, 1)
        return transformed_points.T  # Transpose back to Nx3
    
    def inverse(self) -> 'FrameTransform':
        """
        Compute the inverse transformation.
        
        Returns:
            FrameTransform: New FrameTransform object representing the inverse transformation
        """
        # Inverse transformation: F⁻¹ = [R⁻¹, -R⁻¹ * p]
        # For rotation matrices: R⁻¹ = R.T (transpose)
        inverse_rotation = self.rotation_matrix.T
        inverse_translation = -np.dot(inverse_rotation, self.translation_vector)
        
        return FrameTransform(inverse_rotation, inverse_translation)
    
    def compose(self, other_transform: 'FrameTransform') -> 'FrameTransform':
        """
        Compose this transformation with another transformation.
        
        Args:
            other_transform (FrameTransform): Another transformation to compose with
            
        Returns:
            FrameTransform: New FrameTransform object representing the composition
        """
        # Composition: F₁ • F₂ = [R₁ * R₂, R₁ * p₂ + p₁]
        # This transformation is F₁, other_transform is F₂
        composed_rotation = np.dot(self.rotation_matrix, other_transform.rotation_matrix)
        composed_translation = np.dot(self.rotation_matrix, other_transform.translation_vector) + self.translation_vector
        
        return FrameTransform(composed_rotation, composed_translation)

    @staticmethod
    def Point_set_registration(point_set_1: np.ndarray, point_set_2: np.ndarray, weights: np.ndarray=None) -> 'FrameTransform':
        """
        Register two point sets using the Point Set Registration algorithm.
        
        Args:
            point_set_1 (np.ndarray): First point set (N,3)
            point_set_2 (np.ndarray): Second point set (N,3) 
            weights (np.ndarray, optional): Weight vector (N,) for each correspondence
        """
        src_points = np.asarray(point_set_1, dtype=float)
        dst_points = np.asarray(point_set_2, dtype=float)
        
        assert src_points.shape == dst_points.shape and src_points.ndim == 2 and src_points.shape[1] == 3 and src_points.shape[0] >= 3, "point_set_1/point_set_2 must be (N,3), N>=3"
        
        if weights is None:
            point_weights = np.ones(src_points.shape[0], dtype=float)
        else:
            point_weights = np.asarray(weights, dtype=float).reshape(-1)
            assert point_weights.shape[0] == src_points.shape[0], "weights must have same length as point sets"
        
        # Normalize weights
        normalized_weights = point_weights / point_weights.sum()
        
        # Weighted centroids
        src_centroid = (normalized_weights[:, None] * src_points).sum(axis=0)
        dst_centroid = (normalized_weights[:, None] * dst_points).sum(axis=0)
        
        # Zero-mean (centered points)
        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid
        
        # Weighted cross-covariance matrix
        # H = sum_i w_i * src_centered_i^T * dst_centered_i
        # Implemented as H = (normalized_weights * src_centered)^T @ dst_centered
        cross_covariance = (normalized_weights[:, None] * src_centered).T @ dst_centered  # (3,3)
        
        # Calculate the singular value decomposition of the covariance matrix
        U, singular_values, Vt = np.linalg.svd(cross_covariance)
        
        # Calculate the rotation matrix
        rotation_matrix = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = Vt.T @ U.T
        
        # Calculate the translation vector
        translation_vector = dst_centroid - rotation_matrix @ src_centroid
        
        return FrameTransform(rotation_matrix, translation_vector)