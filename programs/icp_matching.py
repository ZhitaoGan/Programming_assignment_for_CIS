"""
ICP Matching Module for PA3

Implements matching component of Iterative Closest Point (ICP) algorithm.
PA3: Single iteration with F_reg = Identity
PA4: Add iterative refinement loop
"""

import numpy as np
from programs.frame_transform import FrameTransform


def compute_pointer_tip_in_bone_frame(body_A, body_B, frame_data):
    """
    Compute pointer tip position in bone coordinate frame.

    Formula: d_k = F_B^(-1) · F_A · A_tip

    Args:
        body_A (dict): 'markers' (N_A, 3), 'tip' (3,)
        body_B (dict): 'markers' (N_B, 3)
        frame_data (dict): 'a_markers' (N_A, 3), 'b_markers' (N_B, 3)

    Returns:
        dict: 'd_k', 'F_A', 'F_B'
    """
    F_A = FrameTransform.Point_set_registration(
        body_A['markers'],
        frame_data['a_markers']
    )

    F_B = FrameTransform.Point_set_registration(
        body_B['markers'],
        frame_data['b_markers']
    )

    tip_in_tracker = F_A.transform_points(body_A['tip'].reshape(1, -1))[0]
    F_B_inv = F_B.inverse()
    d_k = F_B_inv.transform_points(tip_in_tracker.reshape(1, -1))[0]

    return {
        'd_k': d_k,
        'F_A': F_A,
        'F_B': F_B
    }


def find_closest_point_on_triangle(point, triangle_vertices):
    """
    Find closest point on triangle (lecture slides 11-13).

    Args:
        point (np.ndarray): Query point, shape (3,)
        triangle_vertices (np.ndarray): [p, q, r], shape (3, 3)

    Returns:
        dict: 'closest_point', 'distance'
    """
    p = triangle_vertices[0]
    q = triangle_vertices[1]
    r = triangle_vertices[2]

    u = q - p
    v = r - p
    w = point - p

    u_dot_u = np.dot(u, u)
    u_dot_v = np.dot(u, v)
    v_dot_v = np.dot(v, v)
    w_dot_u = np.dot(w, u)
    w_dot_v = np.dot(w, v)

    denom = u_dot_u * v_dot_v - u_dot_v * u_dot_v

    if abs(denom) < 1e-12:
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

    c = p + lambda_val * u + mu_val * v

    if lambda_val >= 0 and mu_val >= 0 and (lambda_val + mu_val) <= 1:
        distance = np.linalg.norm(point - c)
        return {'closest_point': c, 'distance': distance}

    if lambda_val < 0:
        closest, distance = _project_on_segment(point, r, p)
        return {'closest_point': closest, 'distance': distance}
    elif mu_val < 0:
        closest, distance = _project_on_segment(point, p, q)
        return {'closest_point': closest, 'distance': distance}
    else:
        closest, distance = _project_on_segment(point, q, r)
        return {'closest_point': closest, 'distance': distance}


def _project_on_segment(c, p, q):
    """
    Project point onto segment (lecture slide 13).

    Args:
        c, p, q (np.ndarray): Points, shape (3,)

    Returns:
        tuple: (c_star, distance)
    """
    qp = q - p
    qp_dot_qp = np.dot(qp, qp)

    if qp_dot_qp < 1e-12:
        return p, np.linalg.norm(c - p)

    lambda_val = np.dot(c - p, qp) / qp_dot_qp
    lambda_seg = max(0.0, min(lambda_val, 1.0))
    c_star = p + lambda_seg * qp

    return c_star, np.linalg.norm(c - c_star)


def find_closest_point_on_mesh(point, mesh):
    """
    Find closest point on mesh (linear search).

    Args:
        point (np.ndarray): Query point, shape (3,)
        mesh (dict): 'vertices', 'triangles'

    Returns:
        dict: 'closest_point', 'distance', 'triangle_index'
    """
    vertices = mesh['vertices']
    triangles = mesh['triangles']

    min_distance = float('inf')
    closest_point = None
    closest_triangle_idx = -1

    for tri_idx, triangle_indices in enumerate(triangles):
        triangle_vertices = vertices[triangle_indices]
        result = find_closest_point_on_triangle(point, triangle_vertices)

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
    Process single frame for PA3.

    Args:
        body_A, body_B (dict): Body definitions
        mesh (dict): Surface mesh
        frame_data (dict): Marker positions in tracker coords
        F_reg (FrameTransform, optional): PA3: Identity, PA4: refined

    Returns:
        dict: 'd_k', 's_k', 'c_k', 'distance', 'triangle_index'
    """
    tip_result = compute_pointer_tip_in_bone_frame(body_A, body_B, frame_data)
    d_k = tip_result['d_k']

    if F_reg is None:
        s_k = d_k.copy()
    else:
        s_k = F_reg.transform_points(d_k.reshape(1, -1))[0]

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
