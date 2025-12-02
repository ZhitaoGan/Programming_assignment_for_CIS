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
from programs.frame_transform import FrameTransform


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


class TestICPAlgorithm(unittest.TestCase):
    """Test complete ICP algorithm (PA4)."""

    def test_icp_converges_identity_case(self):
        """Test ICP when data is already registered (should converge to identity)."""
        # Create simple rigid bodies with 3 markers each
        body_A_markers = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        body_A_tip = np.array([0.5, 0.5, 0.5])
        body_A = {
            'N_markers': 3,
            'markers': body_A_markers,
            'tip': body_A_tip
        }

        body_B_markers = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        body_B = {
            'N_markers': 3,
            'markers': body_B_markers
        }

        # Create a simple mesh - single triangle
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        triangles = np.array([[0, 1, 2]])
        mesh = {
            'vertices': vertices,
            'triangles': triangles,
            'N_vertices': 3,
            'N_triangles': 1
        }

        # Create frames where pointer tip points at the mesh
        # Need at least 3 frames for point set registration (requires N >= 3)
        frame1 = {
            'a_markers': body_A_markers.copy(),
            'b_markers': body_B_markers.copy()
        }

        frame2 = {
            'a_markers': body_A_markers + np.array([0.1, 0.1, 0.0]),
            'b_markers': body_B_markers + np.array([0.1, 0.1, 0.0])
        }

        frame3 = {
            'a_markers': body_A_markers + np.array([0.0, 0.1, 0.1]),
            'b_markers': body_B_markers + np.array([0.0, 0.1, 0.1])
        }

        all_frames = [frame1, frame2, frame3]

        # Run ICP
        result = icp_matching.run_icp_on_all_frames(
            body_A, body_B, mesh, all_frames,
            max_iterations=10,
            convergence_threshold=1e-5
        )

        # Check that ICP converged
        self.assertTrue(result['converged'], "ICP should converge")
        self.assertGreater(result['iterations'], 0, "Should run at least one iteration")
        self.assertLessEqual(result['iterations'], 10, "Should not exceed max iterations")

        # Check that F_reg is reasonable (close to identity for this simple case)
        F_reg = result['F_reg']
        self.assertIsInstance(F_reg, FrameTransform, "Should return FrameTransform")

        # F_reg should be close to identity (since tip and mesh are already aligned)
        # Rotation should be near identity matrix (within 0.1)
        rotation_diff = np.linalg.norm(F_reg.rotation_matrix - np.eye(3), 'fro')
        self.assertLess(rotation_diff, 0.2,
                       f"Rotation should be close to identity, but diff={rotation_diff}")
        # Translation should be small
        translation_norm = np.linalg.norm(F_reg.translation_vector)
        self.assertLess(translation_norm, 1.0, "Translation should be small for this simple case")

        # Check results were computed
        self.assertEqual(len(result['results']), 3, "Should have results for 3 frames")

        # Each result should have required fields
        for i, frame_result in enumerate(result['results']):
            self.assertIn('d_k', frame_result)
            self.assertIn('s_k', frame_result)
            self.assertIn('c_k', frame_result)
            self.assertIn('distance', frame_result)

            # All should be numpy arrays of length 3
            self.assertEqual(len(frame_result['d_k']), 3)
            self.assertEqual(len(frame_result['s_k']), 3)
            self.assertEqual(len(frame_result['c_k']), 3)

            # Distance should be non-negative and reasonable
            self.assertGreaterEqual(frame_result['distance'], 0.0)

            # s_k should be on or near the mesh (distance should be small)
            # Since mesh is a single triangle at z=0, the closest point should have z=0
            self.assertAlmostEqual(frame_result['c_k'][2], 0.0, places=5,
                                 msg=f"Frame {i}: closest point should be on mesh plane (z=0)")

            # Distance should be approximately the z-component of s_k (since mesh is at z=0)
            expected_distance = abs(frame_result['s_k'][2])
            self.assertAlmostEqual(frame_result['distance'], expected_distance, places=2,
                                 msg=f"Frame {i}: distance should match z-height above mesh")

    def test_icp_with_translation(self):
        """Test ICP when there's a known translation to recover."""
        # Create bodies
        body_A_markers = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        body_A_tip = np.array([0.0, 0.0, 0.0])
        body_A = {
            'N_markers': 3,
            'markers': body_A_markers,
            'tip': body_A_tip
        }

        body_B_markers = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        body_B = {
            'N_markers': 3,
            'markers': body_B_markers
        }

        # Create mesh at origin
        vertices = np.array([
            [-5.0, -5.0, 0.0],
            [ 5.0, -5.0, 0.0],
            [ 5.0,  5.0, 0.0],
            [-5.0,  5.0, 0.0]
        ])
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        mesh = {
            'vertices': vertices,
            'triangles': triangles,
            'N_vertices': 4,
            'N_triangles': 2
        }

        # Create at least 3 frames (required by point set registration)
        frame1 = {
            'a_markers': body_A_markers.copy(),
            'b_markers': body_B_markers.copy()
        }
        frame2 = {
            'a_markers': body_A_markers + np.array([0.05, 0.0, 0.0]),
            'b_markers': body_B_markers + np.array([0.05, 0.0, 0.0])
        }
        frame3 = {
            'a_markers': body_A_markers + np.array([0.0, 0.05, 0.0]),
            'b_markers': body_B_markers + np.array([0.0, 0.05, 0.0])
        }
        all_frames = [frame1, frame2, frame3]

        # Run ICP
        result = icp_matching.run_icp_on_all_frames(
            body_A, body_B, mesh, all_frames,
            max_iterations=50,
            convergence_threshold=1e-6
        )

        # Check convergence
        self.assertTrue(result['converged'] or result['iterations'] > 0,
                       "ICP should run at least one iteration")

        # Check output structure
        self.assertEqual(len(result['results']), 3)
        self.assertIn('F_reg', result)
        self.assertIn('iterations', result)

        # Verify all distances are reasonable (should be close to mesh)
        for i, frame_result in enumerate(result['results']):
            # Distance should be non-negative
            self.assertGreaterEqual(frame_result['distance'], 0.0,
                                  msg=f"Frame {i}: distance should be non-negative")

            # Since mesh is large (10x10) and at z=0, and tip is at origin,
            # the closest point should be at z=0
            self.assertAlmostEqual(frame_result['c_k'][2], 0.0, places=5,
                                 msg=f"Frame {i}: closest point should be on mesh plane")

            # After ICP convergence, s_k should be close to c_k
            distance_computed = np.linalg.norm(frame_result['s_k'] - frame_result['c_k'])
            self.assertAlmostEqual(frame_result['distance'], distance_computed, places=5,
                                 msg=f"Frame {i}: distance should equal ||s_k - c_k||")


if __name__ == '__main__':
    unittest.main()
