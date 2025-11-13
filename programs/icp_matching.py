"""
ICP Matching Module for PA3

This module contains functions for Iterative Closest Point (ICP) matching algorithm.
For PA3, we implement the matching part of ICP (finding closest points on a surface mesh).
For PA4, we will add the iterative refinement loop.

Key Functions:
- compute_pointer_tip_in_bone_frame: Compute pointer tip position in bone frame (Steps 2+3)
- find_closest_point_on_triangle: Find closest point on a single triangle
- find_closest_point_on_mesh: Find closest point on entire mesh (linear search)
"""

import numpy as np
from programs.frame_transform import FrameTransform


def compute_pointer_tip_in_bone_frame(body_A, body_B, frame_data):
    """
    Compute pointer tip position in bone coordinate frame (PA3 Steps 2+3).

    This function:
    1. Computes F_A: transformation from body A to tracker coordinates
    2. Computes F_B: transformation from body B to tracker coordinates
    3. Computes d_k: pointer tip position in body B frame

    The formula is: d_k = F_B^(-1) · F_A · A_tip

    Args:
        body_A (dict): Body A definition containing:
            - 'markers': LED marker positions in body A coordinates, shape (N_A, 3)
            - 'tip': Tip position in body A coordinates, shape (3,)
        body_B (dict): Body B definition containing:
            - 'markers': LED marker positions in body B coordinates, shape (N_B, 3)
        frame_data (dict): Frame data containing:
            - 'a_markers': Body A LED positions in tracker coordinates, shape (N_A, 3)
            - 'b_markers': Body B LED positions in tracker coordinates, shape (N_B, 3)

    Returns:
        dict: Dictionary containing:
            - 'd_k': Pointer tip position in body B frame, shape (3,)
            - 'F_A': FrameTransform object for body A to tracker
            - 'F_B': FrameTransform object for body B to tracker
    """
    # Step 2a: Compute F_A (body A to tracker)
    # Input: A_i (body coords), {a_i,k} (tracker coords)
    # Output: F_A = [R_A | p_A]
    F_A = FrameTransform.Point_set_registration(
        body_A['markers'],  # Source: body A coordinates
        frame_data['a_markers']  # Target: tracker coordinates
    )

    # Step 2b: Compute F_B (body B to tracker)
    # Input: B_i (body coords), {b_i,k} (tracker coords)
    # Output: F_B = [R_B | p_B]
    F_B = FrameTransform.Point_set_registration(
        body_B['markers'],  # Source: body B coordinates
        frame_data['b_markers']  # Target: tracker coordinates
    )

    # Step 3: Compute pointer tip position in body B frame
    # d_k = F_B^(-1) · F_A · A_tip

    # First, transform A_tip to tracker coordinates
    tip_in_tracker = F_A.transform_points(body_A['tip'].reshape(1, -1))[0]

    # Then, transform from tracker to body B coordinates
    F_B_inv = F_B.inverse()
    d_k = F_B_inv.transform_points(tip_in_tracker.reshape(1, -1))[0]

    return {
        'd_k': d_k,
        'F_A': F_A,
        'F_B': F_B
    }


def find_closest_point_on_triangle(point, triangle_vertices):
    """
    Find the closest point on a triangle to a given point.

    This implements the algorithm from lecture slides 11-13:
    1. Solve least squares: a - p ≈ λ(q-p) + μ(r-p)
    2. Compute c = p + λ(q-p) + μ(r-p)
    3. Check if inside triangle: λ ≥ 0 AND μ ≥ 0 AND λ+μ ≤ 1
    4. Otherwise project onto edges

    Args:
        point (np.ndarray): Query point 'a', shape (3,)
        triangle_vertices (np.ndarray): Triangle vertices [p, q, r], shape (3, 3)

    Returns:
        dict: Dictionary containing:
            - 'closest_point': Closest point on triangle, shape (3,)
            - 'distance': Distance from query point to closest point (scalar)
    """
    p = triangle_vertices[0]  # Vertex p
    q = triangle_vertices[1]  # Vertex q
    r = triangle_vertices[2]  # Vertex r

    # Step 1: Solve least squares for λ, μ
    # Set u = q-p, v = r-p, w = a-p
    u = q - p
    v = r - p
    w = point - p

    # Solve 2×2 system for λ, μ
    # [u·u  u·v] [λ]   [w·u]
    # [u·v  v·v] [μ] = [w·v]

    u_dot_u = np.dot(u, u)
    u_dot_v = np.dot(u, v)
    v_dot_v = np.dot(v, v)
    w_dot_u = np.dot(w, u)
    w_dot_v = np.dot(w, v)

    denom = u_dot_u * v_dot_v - u_dot_v * u_dot_v

    # Handle degenerate triangle
    if abs(denom) < 1e-12:
        # Degenerate triangle, project onto edges
        closest_on_pq, dist_pq = _project_on_segment(point, p, q)
        closest_on_qr, dist_qr = _project_on_segment(point, q, r)
        closest_on_rp, dist_rp = _project_on_segment(point, r, p)

        min_dist = min(dist_pq, dist_qr, dist_rp)
        if min_dist == dist_pq:
            return {'closest_point': closest_on_pq, 'distance': dist_pq}
        elif min_dist == dist_qr:
            return {'closest_point': closest_on_qr, 'distance': dist_qr}
        else:
            return {'closest_point': closest_on_rp, 'distance': dist_rp}

    lambda_val = (v_dot_v * w_dot_u - u_dot_v * w_dot_v) / denom
    mu_val = (u_dot_u * w_dot_v - u_dot_v * w_dot_u) / denom

    # Step 2: Compute c = p + λ(q-p) + μ(r-p)
    c = p + lambda_val * u + mu_val * v

    # Step 3: Check if inside triangle
    if lambda_val >= 0 and mu_val >= 0 and (lambda_val + mu_val) <= 1:
        distance = np.linalg.norm(point - c)
        return {'closest_point': c, 'distance': distance}

    # Step 4: Otherwise project onto edges
    # If λ < 0: ProjectOnSegment(a, r, p)
    # If μ < 0: ProjectOnSegment(a, p, q)
    # If λ+μ > 1: ProjectOnSegment(a, q, r)

    if lambda_val < 0:
        closest, distance = _project_on_segment(point, r, p)
        return {'closest_point': closest, 'distance': distance}
    elif mu_val < 0:
        closest, distance = _project_on_segment(point, p, q)
        return {'closest_point': closest, 'distance': distance}
    else:  # lambda_val + mu_val > 1
        closest, distance = _project_on_segment(point, q, r)
        return {'closest_point': closest, 'distance': distance}


def _project_on_segment(c, p, q):
    """
    Project point onto a line segment (from lecture slide 13).

    Algorithm:
    λ = (c-p)·(q-p) / (q-p)·(q-p)
    λ_seg = Max(0, Min(λ, 1))
    c* = p + λ_seg × (q-p)

    Args:
        c (np.ndarray): Query point, shape (3,)
        p (np.ndarray): Segment start point, shape (3,)
        q (np.ndarray): Segment end point, shape (3,)

    Returns:
        tuple: (c_star, distance)
            - c_star: Closest point on segment, shape (3,)
            - distance: Distance from query point to closest point (scalar)
    """
    qp = q - p
    qp_dot_qp = np.dot(qp, qp)

    # Handle degenerate segment (p == q)
    if qp_dot_qp < 1e-12:
        distance = np.linalg.norm(c - p)
        return p, distance

    # λ = (c-p)·(q-p) / (q-p)·(q-p)
    lambda_val = np.dot(c - p, qp) / qp_dot_qp

    # λ_seg = Max(0, Min(λ, 1))
    lambda_seg = max(0.0, min(lambda_val, 1.0))

    # c* = p + λ_seg × (q-p)
    c_star = p + lambda_seg * qp

    distance = np.linalg.norm(c - c_star)

    return c_star, distance


def find_closest_point_on_mesh(point, mesh):
    """
    Find the closest point on a surface mesh to a given point (linear search).

    This is a simple brute-force implementation that searches all triangles.
    For PA3, this is acceptable. For larger meshes, you could optimize with:
    - Bounding sphere tree
    - K-d tree
    - Octree

    Args:
        point (np.ndarray): Query point, shape (3,)
        mesh (dict): Surface mesh containing:
            - 'vertices': Vertex positions, shape (N_vertices, 3)
            - 'triangles': Triangle vertex indices, shape (N_triangles, 3)

    Returns:
        dict: Dictionary containing:
            - 'closest_point': Closest point on mesh, shape (3,)
            - 'distance': Distance from query point to closest point (scalar)
            - 'triangle_index': Index of the closest triangle
    """
    vertices = mesh['vertices']
    triangles = mesh['triangles']

    min_distance = float('inf')
    closest_point = None
    closest_triangle_idx = -1

    # Linear search through all triangles
    for tri_idx, triangle_indices in enumerate(triangles):
        # Get the three vertices of this triangle
        triangle_vertices = vertices[triangle_indices]

        # Find closest point on this triangle
        result = find_closest_point_on_triangle(point, triangle_vertices)

        # Update if this is closer
        if result['distance'] < min_distance:
            min_distance = result['distance']
            closest_point = result['closest_point']
            closest_triangle_idx = tri_idx

    return {
        'closest_point': closest_point,
        'distance': min_distance,
        'triangle_index': closest_triangle_idx
    }


def process_pa3_frame(body_A, body_B, mesh, frame_data, F_reg=None):
    """
    Process a single frame for PA3 (complete pipeline for one sample).

    This function:
    1. Computes pointer tip in bone frame (d_k)
    2. Applies F_reg transformation if provided (for PA3, F_reg = Identity)
    3. Finds closest point on mesh (c_k)
    4. Computes distance ||d_k - c_k||

    Args:
        body_A (dict): Body A definition (markers + tip)
        body_B (dict): Body B definition (markers)
        mesh (dict): Surface mesh (vertices + triangles)
        frame_data (dict): Frame data (a_markers, b_markers in tracker coords)
        F_reg (FrameTransform, optional): Registration transform (default: Identity)

    Returns:
        dict: Dictionary containing:
            - 'd_k': Pointer tip in bone/body B frame, shape (3,)
            - 's_k': Sample point after F_reg transform, shape (3,)
            - 'c_k': Closest point on mesh, shape (3,)
            - 'distance': ||d_k - c_k|| or ||s_k - c_k|| depending on F_reg
    """
    # Step 1: Compute pointer tip in bone frame (Steps 2+3)
    tip_result = compute_pointer_tip_in_bone_frame(body_A, body_B, frame_data)
    d_k = tip_result['d_k']

    # Step 2: Apply F_reg transformation (for PA3, this is identity)
    if F_reg is None:
        # PA3: F_reg = Identity, so s_k = d_k
        s_k = d_k.copy()
    else:
        # PA4: Apply F_reg transformation
        s_k = F_reg.transform_points(d_k.reshape(1, -1))[0]

    # Step 3: Find closest point on mesh
    mesh_result = find_closest_point_on_mesh(s_k, mesh)
    c_k = mesh_result['closest_point']
    distance = mesh_result['distance']

    return {
        'd_k': d_k,
        's_k': s_k,
        'c_k': c_k,
        'distance': distance,
        'triangle_index': mesh_result['triangle_index']
    }
