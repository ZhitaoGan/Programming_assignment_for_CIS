"""
Unit tests for ICP Matching algorithms (PA3)

Tests the core geometric algorithms:
- Find closest point on triangle
- Find closest point on mesh
- Rigid body tracking
"""

import unittest
import numpy as np
from programs import icp_matching


class TestClosestPointOnTriangle(unittest.TestCase):
    """Test closest point on triangle algorithm."""

    def test_point_inside_triangle(self):
        """Test when point projects inside triangle."""
        # Equilateral triangle in xy-plane
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        r = np.array([0.5, np.sqrt(3)/2, 0.0])
        triangle = np.array([p, q, r])

        # Point directly above center of triangle
        center = (p + q + r) / 3
        query = center + np.array([0.0, 0.0, 1.0])

        result = icp_matching.find_closest_point_on_triangle(query, triangle)

        # Closest point should be the center of triangle
        np.testing.assert_array_almost_equal(result['closest_point'], center, decimal=5)
        # Distance should be 1.0 (height above triangle)
        self.assertAlmostEqual(result['distance'], 1.0, places=5)

    def test_point_on_triangle_vertex(self):
        """Test when point is on a triangle vertex."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        r = np.array([0.0, 1.0, 0.0])
        triangle = np.array([p, q, r])

        # Query point exactly on vertex p
        query = p.copy()

        result = icp_matching.find_closest_point_on_triangle(query, triangle)

        np.testing.assert_array_almost_equal(result['closest_point'], p, decimal=5)
        self.assertAlmostEqual(result['distance'], 0.0, places=5)

    def test_point_on_triangle_edge(self):
        """Test when point projects onto triangle edge."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        r = np.array([0.0, 1.0, 0.0])
        triangle = np.array([p, q, r])

        # Point above midpoint of edge pq
        midpoint = (p + q) / 2
        query = midpoint + np.array([0.0, 0.0, 1.0])

        result = icp_matching.find_closest_point_on_triangle(query, triangle)

        # Should project onto edge midpoint
        np.testing.assert_array_almost_equal(result['closest_point'], midpoint, decimal=5)
        self.assertAlmostEqual(result['distance'], 1.0, places=5)

    def test_point_outside_triangle_near_vertex(self):
        """Test when point is outside triangle, closest to a vertex."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        r = np.array([0.0, 1.0, 0.0])
        triangle = np.array([p, q, r])

        # Point outside triangle, nearest to p
        query = np.array([-1.0, -1.0, 0.0])

        result = icp_matching.find_closest_point_on_triangle(query, triangle)

        # Closest point should be vertex p
        np.testing.assert_array_almost_equal(result['closest_point'], p, decimal=5)
        expected_distance = np.linalg.norm(query - p)
        self.assertAlmostEqual(result['distance'], expected_distance, places=5)

    def test_point_in_triangle_plane(self):
        """Test when point is in the plane of the triangle."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        r = np.array([0.0, 1.0, 0.0])
        triangle = np.array([p, q, r])

        # Point inside triangle, in same plane
        query = np.array([0.25, 0.25, 0.0])

        result = icp_matching.find_closest_point_on_triangle(query, triangle)

        # Closest point should be the query point itself
        np.testing.assert_array_almost_equal(result['closest_point'], query, decimal=5)
        self.assertAlmostEqual(result['distance'], 0.0, places=5)


class TestProjectOnSegment(unittest.TestCase):
    """Test project on segment helper function."""

    def test_project_on_segment_midpoint(self):
        """Test projection onto segment midpoint."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        query = np.array([0.5, 1.0, 0.0])

        closest, distance = icp_matching._project_on_segment(query, p, q)

        expected_closest = np.array([0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(closest, expected_closest, decimal=5)
        self.assertAlmostEqual(distance, 1.0, places=5)

    def test_project_on_segment_endpoint(self):
        """Test projection clamped to segment endpoint."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        query = np.array([2.0, 1.0, 0.0])

        closest, distance = icp_matching._project_on_segment(query, p, q)

        # Should clamp to endpoint q
        np.testing.assert_array_almost_equal(closest, q, decimal=5)
        expected_distance = np.linalg.norm(query - q)
        self.assertAlmostEqual(distance, expected_distance, places=5)

    def test_project_on_point(self):
        """Test projection when segment is degenerate (point)."""
        p = np.array([1.0, 1.0, 1.0])
        q = np.array([1.0, 1.0, 1.0])
        query = np.array([2.0, 2.0, 2.0])

        closest, distance = icp_matching._project_on_segment(query, p, q)

        np.testing.assert_array_almost_equal(closest, p, decimal=5)
        expected_distance = np.linalg.norm(query - p)
        self.assertAlmostEqual(distance, expected_distance, places=5)


class TestClosestPointOnMesh(unittest.TestCase):
    """Test closest point on mesh algorithm."""

    def test_closest_point_on_simple_mesh(self):
        """Test with a simple two-triangle mesh."""
        # Create a simple mesh: two triangles forming a square
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        triangles = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]   # Second triangle
        ])
        mesh = {
            'vertices': vertices,
            'triangles': triangles
        }

        # Query point above center of square
        query = np.array([0.5, 0.5, 1.0])

        result = icp_matching.find_closest_point_on_mesh(query, mesh)

        # Closest point should be at (0.5, 0.5, 0.0)
        expected_closest = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result['closest_point'], expected_closest, decimal=5)
        self.assertAlmostEqual(result['distance'], 1.0, places=5)

    def test_closest_point_on_mesh_vertex(self):
        """Test when closest point is a mesh vertex."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        triangles = np.array([[0, 1, 2]])
        mesh = {
            'vertices': vertices,
            'triangles': triangles
        }

        # Query point near vertex 0
        query = np.array([-0.5, -0.5, 0.0])

        result = icp_matching.find_closest_point_on_mesh(query, mesh)

        # Closest point should be vertex 0
        np.testing.assert_array_almost_equal(result['closest_point'], vertices[0], decimal=5)

    def test_closest_point_single_triangle(self):
        """Test mesh with single triangle."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])
        triangles = np.array([[0, 1, 2]])
        mesh = {
            'vertices': vertices,
            'triangles': triangles
        }

        # Query point above triangle center
        center = vertices.mean(axis=0)
        query = center + np.array([0.0, 0.0, 2.0])

        result = icp_matching.find_closest_point_on_mesh(query, mesh)

        # Distance should be 2.0
        self.assertAlmostEqual(result['distance'], 2.0, places=5)
        # Closest point should be near center
        np.testing.assert_array_almost_equal(result['closest_point'][:2], center[:2], decimal=5)


if __name__ == '__main__':
    unittest.main()
