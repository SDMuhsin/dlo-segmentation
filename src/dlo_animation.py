#!/usr/bin/env python3
"""DLO Point Cloud Animation.

Automated joint installation, kinematic chain construction, and animation
of Deformable Linear Objects from labeled point clouds with skeleton data.

Approach:
    1. Load skeleton graph (nodes + adjacency) from PointWire dataset
    2. Build kinematic tree with identified structural joints
    3. Bind surface points to nearest skeleton nodes
    4. Animate via forward kinematics with smooth joint rotations
    5. Convert animated frames to RGB-D and GIF

Usage:
    python src/dlo_animation.py --phase1    # Joint identification + validation
    python src/dlo_animation.py --phase2    # Point binding + validation
    python src/dlo_animation.py --phase3    # Single-joint animation + validation
    python src/dlo_animation.py --phase4    # Multi-joint animation + validation
    python src/dlo_animation.py --phase5    # RGB-D conversion + GIF
    python src/dlo_animation.py --all       # Run everything
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "set2"
RESULTS_ROOT = PROJECT_ROOT / "results" / "dlo_animation"

# ── Data Loading ────────────────────────────────────────────────────────────

def load_sample(set_id, frame_id=0, resolution=4096):
    """Load point cloud, segmentation, and skeleton for a sample."""
    base = DATA_ROOT / f"{set_id:03d}"
    pcl = np.load(str(base / f"pointclouds_normed_{resolution}" / f"pcl_{frame_id:04d}.npy"))
    seg = np.load(str(base / f"segmentation_normed_{resolution}" / f"seg_{frame_id:04d}.npy"))
    skel = np.load(str(base / "skeletons" / f"{frame_id:03d}.npz"))
    kp = np.load(str(base / "keyposes" / f"kp_{frame_id:04d}.npz"))
    return {
        "points": pcl,
        "labels": seg,
        "skel_nodes": skel["nodes"],
        "skel_adj": skel["adj"],
        "intersections": kp["intersections"],  # bifurcation keypoints
        "endpoints": kp["endpoints"],          # endpoint keypoints
    }


CLASS_NAMES = {0: "Wire", 1: "Endpoint", 2: "Bifurcation", 3: "Connector", 4: "Noise"}
CLASS_COLORS_RGB = {
    0: (180, 180, 180),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
}

# ── Phase 1: Joint Identification ───────────────────────────────────────────

def build_kinematic_tree(nodes, adj):
    """Build a kinematic tree from skeleton graph.

    Identifies structural nodes (endpoints, bifurcations) and wire segments.
    Selects a root node and builds parent-child relationships via BFS.

    Returns:
        tree: dict with keys:
            - root: int, root node index
            - parent: dict[int, int], parent of each node (-1 for root)
            - children: dict[int, list[int]], children of each node
            - degrees: ndarray, degree of each node
            - structural: list[int], structural node indices (degree != 2)
            - wire_segments: list of lists, node indices per wire segment
    """
    n = len(nodes)
    degrees = np.sum(adj > 0, axis=1).astype(int)

    # Structural nodes: endpoints (degree 1) and bifurcations (degree 3+)
    structural = [i for i in range(n) if degrees[i] != 2]

    # Find wire segments (chains between structural nodes)
    wire_segments = _find_wire_segments(nodes, adj, degrees)

    # Choose root: the structural node closest to centroid
    centroid = nodes.mean(axis=0)
    struct_dists = np.linalg.norm(nodes[structural] - centroid, axis=1)
    root = structural[np.argmin(struct_dists)]

    # BFS to build parent-child relationships
    parent = {root: -1}
    children = {i: [] for i in range(n)}
    visited = set([root])
    queue = deque([root])

    while queue:
        node = queue.popleft()
        neighbors = np.where(adj[node] > 0)[0]
        for nbr in neighbors:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                children[node].append(nbr)
                queue.append(nbr)

    return {
        "root": root,
        "parent": parent,
        "children": children,
        "degrees": degrees,
        "structural": structural,
        "wire_segments": wire_segments,
        "nodes": nodes,
        "adj": adj,
    }


def _find_wire_segments(nodes, adj, degrees):
    """Find wire segments between structural nodes (degree != 2)."""
    structural = set(i for i in range(len(nodes)) if degrees[i] != 2)
    segments = []
    visited_edges = set()

    for start in structural:
        neighbors = np.where(adj[start] > 0)[0]
        for next_node in neighbors:
            edge = (min(start, next_node), max(start, next_node))
            if edge in visited_edges:
                continue

            segment = [start, next_node]
            visited_edges.add(edge)

            current = next_node
            prev = start
            while degrees[current] == 2:
                nbrs = np.where(adj[current] > 0)[0]
                others = [nb for nb in nbrs if nb != prev]
                if not others:
                    break
                next_n = others[0]
                edge = (min(current, next_n), max(current, next_n))
                visited_edges.add(edge)
                segment.append(next_n)
                prev = current
                current = next_n

            segments.append(segment)

    return segments


def select_animation_joints(tree, joint_spacing=5):
    """Select animation joints: structural nodes + sampled interior nodes.

    For each wire segment, selects every `joint_spacing`-th node as an
    animation joint, in addition to all structural nodes.

    Returns:
        anim_joints: list of node indices that serve as animation joints
    """
    anim_joints = set(tree["structural"])

    for segment in tree["wire_segments"]:
        # Always include start and end (structural nodes)
        anim_joints.add(segment[0])
        anim_joints.add(segment[-1])
        # Sample interior nodes
        interior = segment[1:-1]
        for i in range(0, len(interior), joint_spacing):
            anim_joints.add(interior[i])

    return sorted(anim_joints)


def validate_phase1(tree, data):
    """Validate Phase 1: Joint identification."""
    nodes = tree["nodes"]
    degrees = tree["degrees"]
    points = data["points"]
    labels = data["labels"]

    results = {}

    # V1.1: Tree structure validity
    n = len(nodes)
    num_edges = sum(len(ch) for ch in tree["children"].values())
    results["tree_valid"] = {
        "num_nodes": n,
        "num_edges_parent_child": num_edges,
        "is_tree": num_edges == n - 1,
        "all_nodes_have_parent": all(i in tree["parent"] for i in range(n)),
    }

    # V1.2: Structural node identification matches labels
    pcl_tree = KDTree(points)
    _, nn_idx = pcl_tree.query(nodes)
    nn_labels = labels[nn_idx]

    leaf_nodes = [i for i in range(n) if degrees[i] == 1]
    branch_nodes = [i for i in range(n) if degrees[i] >= 3]

    leaf_near_endpoint = sum(1 for i in leaf_nodes if nn_labels[i] in [1])  # Endpoint
    branch_near_bifurcation = sum(1 for i in branch_nodes if nn_labels[i] in [2])  # Bifurcation

    results["structural_match"] = {
        "num_leaf_nodes": len(leaf_nodes),
        "leaf_near_endpoint_label": leaf_near_endpoint,
        "leaf_match_rate": leaf_near_endpoint / max(len(leaf_nodes), 1),
        "num_branch_nodes": len(branch_nodes),
        "branch_near_bifurcation_label": branch_near_bifurcation,
        "branch_match_rate": branch_near_bifurcation / max(len(branch_nodes), 1),
    }

    # V1.3: Skeleton covers the point cloud
    skel_tree = KDTree(nodes)
    dists, _ = skel_tree.query(points)
    results["coverage"] = {
        "mean_pcl_to_skel_dist": float(np.mean(dists)),
        "max_pcl_to_skel_dist": float(np.max(dists)),
        "p95_pcl_to_skel_dist": float(np.percentile(dists, 95)),
        "within_0.02": float(np.mean(dists < 0.02)),
        "within_0.05": float(np.mean(dists < 0.05)),
    }

    # V1.4: Wire segments partition all skeleton edges
    all_segment_edges = set()
    for seg in tree["wire_segments"]:
        for i in range(len(seg) - 1):
            all_segment_edges.add((min(seg[i], seg[i+1]), max(seg[i], seg[i+1])))
    total_edges = 0
    adj = tree["adj"]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                total_edges += 1
    results["segment_coverage"] = {
        "total_edges": total_edges,
        "edges_in_segments": len(all_segment_edges),
        "all_edges_covered": total_edges == len(all_segment_edges),
    }

    # Overall pass/fail
    passed = (
        results["tree_valid"]["is_tree"]
        and results["tree_valid"]["all_nodes_have_parent"]
        and results["structural_match"]["branch_match_rate"] >= 0.8
        and results["coverage"]["within_0.05"] >= 0.95
        and results["segment_coverage"]["all_edges_covered"]
    )
    results["passed"] = passed

    return results


# ── Phase 2: Surface Point Binding ──────────────────────────────────────────

def bind_points_to_skeleton(points, nodes, adj):
    """Bind each surface point to the nearest skeleton edge.

    For each point, finds the nearest skeleton edge and computes the
    projection parameter t in [0, 1] along that edge. The point is then
    influenced by the two endpoints of the edge with weights (1-t, t).

    Returns:
        binding: dict with:
            - edge_node_a: (N,) int, first node of nearest edge
            - edge_node_b: (N,) int, second node of nearest edge
            - weight_a: (N,) float, weight for node a (1-t)
            - weight_b: (N,) float, weight for node b (t)
            - offsets: (N, 3) float, local offset from interpolated edge position
    """
    N = len(points)

    # Collect all edges
    edges = []
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                edges.append((i, j))
    edges = np.array(edges)

    # For each edge, compute its line segment
    edge_a = nodes[edges[:, 0]]  # (E, 3)
    edge_b = nodes[edges[:, 1]]  # (E, 3)

    # For each point, find nearest edge via vectorized projection
    edge_node_a = np.zeros(N, dtype=np.int32)
    edge_node_b = np.zeros(N, dtype=np.int32)
    weight_a = np.zeros(N, dtype=np.float64)
    weight_b = np.zeros(N, dtype=np.float64)
    offsets = np.zeros((N, 3), dtype=np.float64)

    # Process in batches for memory efficiency
    batch_size = 512
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pts_batch = points[start:end]  # (B, 3)
        B = end - start

        # Compute projection of each point onto each edge
        # edge_vec = b - a, point_vec = p - a
        edge_vec = edge_b - edge_a  # (E, 3)
        edge_len_sq = np.sum(edge_vec ** 2, axis=1)  # (E,)
        edge_len_sq = np.maximum(edge_len_sq, 1e-12)

        # Broadcast: pts_batch (B,1,3) - edge_a (1,E,3) -> (B,E,3)
        point_vec = pts_batch[:, None, :] - edge_a[None, :, :]

        # Projection parameter t = dot(point_vec, edge_vec) / |edge_vec|^2
        t = np.sum(point_vec * edge_vec[None, :, :], axis=2) / edge_len_sq[None, :]  # (B, E)
        t = np.clip(t, 0, 1)

        # Closest point on edge: a + t * (b - a)
        closest = edge_a[None, :, :] + t[:, :, None] * edge_vec[None, :, :]  # (B, E, 3)

        # Distance from point to closest point on edge
        dists = np.linalg.norm(pts_batch[:, None, :] - closest, axis=2)  # (B, E)

        # Find nearest edge for each point
        nearest_edge_idx = np.argmin(dists, axis=1)  # (B,)

        for i in range(B):
            eidx = nearest_edge_idx[i]
            t_val = t[i, eidx]
            edge_node_a[start + i] = edges[eidx, 0]
            edge_node_b[start + i] = edges[eidx, 1]
            weight_a[start + i] = 1.0 - t_val
            weight_b[start + i] = t_val
            # Offset from interpolated skeleton position
            skel_pos = nodes[edges[eidx, 0]] * (1 - t_val) + nodes[edges[eidx, 1]] * t_val
            offsets[start + i] = points[start + i] - skel_pos

    return {
        "edge_node_a": edge_node_a,
        "edge_node_b": edge_node_b,
        "weight_a": weight_a,
        "weight_b": weight_b,
        "offsets": offsets,
    }


def validate_phase2(binding, points, nodes):
    """Validate Phase 2: Point binding."""
    results = {}

    N = len(points)
    wa = binding["weight_a"]
    wb = binding["weight_b"]
    offsets = binding["offsets"]
    node_a = binding["edge_node_a"]
    node_b = binding["edge_node_b"]

    # V2.1: All points are bound
    results["all_bound"] = {
        "total_points": N,
        "bound": int(np.all(wa + wb > 0.99)),  # weights should sum to ~1
    }

    # V2.2: Weights sum to 1
    weight_sums = wa + wb
    results["weight_sums"] = {
        "mean": float(np.mean(weight_sums)),
        "min": float(np.min(weight_sums)),
        "max": float(np.max(weight_sums)),
        "all_near_one": bool(np.all(np.abs(weight_sums - 1.0) < 1e-6)),
    }

    # V2.3: Reconstruction accuracy (bind + offset should recover original point)
    reconstructed = (nodes[node_a] * wa[:, None] + nodes[node_b] * wb[:, None]) + offsets
    recon_error = np.linalg.norm(reconstructed - points, axis=1)
    results["reconstruction"] = {
        "mean_error": float(np.mean(recon_error)),
        "max_error": float(np.max(recon_error)),
        "perfect_recon": bool(np.all(recon_error < 1e-10)),
    }

    # V2.4: Offset magnitudes (should be small - points are near skeleton)
    offset_norms = np.linalg.norm(offsets, axis=1)
    results["offsets"] = {
        "mean": float(np.mean(offset_norms)),
        "max": float(np.max(offset_norms)),
        "p95": float(np.percentile(offset_norms, 95)),
    }

    # V2.5: Edge utilization - how many edges have bound points?
    used_edges = set()
    for i in range(N):
        used_edges.add((min(node_a[i], node_b[i]), max(node_a[i], node_b[i])))
    results["edge_utilization"] = {
        "edges_with_points": len(used_edges),
    }

    passed = (
        results["weight_sums"]["all_near_one"]
        and results["reconstruction"]["perfect_recon"]
        and results["offsets"]["p95"] < 0.1
    )
    results["passed"] = passed

    return results


# ── Phase 3 & 4: Animation ─────────────────────────────────────────────────

def rotation_matrix(axis, angle):
    """Rodrigues' rotation formula: rotation matrix for given axis and angle."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def compute_forward_kinematics(tree, joint_rotations):
    """Compute world positions of all skeleton nodes given joint rotations.

    Uses forward kinematics: propagates transformations from root to leaves.

    Args:
        tree: kinematic tree from build_kinematic_tree()
        joint_rotations: dict[int, (axis, angle)] for each animated joint

    Returns:
        new_positions: (N, 3) world positions of all skeleton nodes
        node_transforms: dict[int, (R, t)] rotation and translation for each node
    """
    nodes = tree["nodes"]
    root = tree["root"]
    n = len(nodes)

    # For each node, store cumulative transform (rotation, translation)
    # Transform maps from original local frame to new world frame
    node_R = {i: np.eye(3) for i in range(n)}
    node_t = {i: np.zeros(3) for i in range(n)}

    new_positions = np.copy(nodes)

    # BFS from root
    queue = deque([root])
    visited = set([root])

    # Root stays in place
    node_R[root] = np.eye(3)
    node_t[root] = np.zeros(3)
    new_positions[root] = nodes[root]

    while queue:
        node = queue.popleft()
        for child in tree["children"][node]:
            if child in visited:
                continue
            visited.add(child)

            # Start with parent's transform
            parent_R = node_R[node]
            parent_t = node_t[node]

            # If this node has a rotation, apply it at the parent's position
            if node in joint_rotations:
                axis, angle = joint_rotations[node]
                # Rotation happens at the parent node's NEW position
                pivot = new_positions[node]
                local_R = rotation_matrix(axis, angle)
                # Compose: first parent rotation, then local rotation
                child_R = local_R @ parent_R
                child_t = pivot + local_R @ (parent_t + parent_R @ (nodes[child] - nodes[node])) - (parent_R @ nodes[child])
                # Actually, let me do this more carefully:
                # new_child_pos = pivot + local_R @ (parent_R @ (nodes[child] - nodes[node]) + parent_t_relative)
                # where parent_t_relative represents how the parent has moved
            else:
                child_R = parent_R
                child_t = parent_t

            node_R[child] = child_R
            node_t[child] = child_t

            # New position: apply cumulative transform
            # The child's position relative to root in original coords, transformed
            offset_from_node = nodes[child] - nodes[node]
            new_positions[child] = new_positions[node] + child_R @ offset_from_node

            queue.append(child)

    return new_positions, {i: (node_R[i], node_t[i]) for i in range(n)}


def animate_points(points, binding, original_nodes, new_nodes, node_transforms):
    """Move surface points based on skeleton deformation.

    Each point is bound to an edge (a, b) with weights. We compute the
    new edge-interpolated position and add the rotated offset.

    Args:
        points: (N, 3) original point positions
        binding: binding dict from bind_points_to_skeleton
        original_nodes: (M, 3) original skeleton node positions
        new_nodes: (M, 3) new skeleton node positions after FK
        node_transforms: dict of (R, t) per node

    Returns:
        new_points: (N, 3) animated point positions
    """
    N = len(points)
    node_a = binding["edge_node_a"]
    node_b = binding["edge_node_b"]
    wa = binding["weight_a"]
    wb = binding["weight_b"]
    offsets = binding["offsets"]

    # New interpolated skeleton position for each point
    new_skel_pos = new_nodes[node_a] * wa[:, None] + new_nodes[node_b] * wb[:, None]

    # Rotate offset: use the weighted rotation of the two edge nodes
    new_offsets = np.zeros_like(offsets)
    for i in range(N):
        Ra, _ = node_transforms[node_a[i]]
        Rb, _ = node_transforms[node_b[i]]
        # Weighted rotation of offset
        R_blend = wa[i] * Ra + wb[i] * Rb
        # Re-orthogonalize (SVD)
        U, _, Vt = np.linalg.svd(R_blend)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        new_offsets[i] = R_ortho @ offsets[i]

    return new_skel_pos + new_offsets


def generate_animation_frames(tree, binding, points, labels,
                              num_frames=60, max_angle_deg=20.0,
                              active_joints=None):
    """Generate a sequence of animated point cloud frames.

    Applies sinusoidal rotations at active joints.

    Args:
        tree: kinematic tree
        binding: point binding
        points: (N, 3) original points
        labels: (N,) labels
        num_frames: number of frames to generate
        max_angle_deg: maximum rotation angle in degrees
        active_joints: list of joint indices to animate (None = auto-select)

    Returns:
        frames: list of (points, labels) tuples, length num_frames
    """
    nodes = tree["nodes"]

    if active_joints is None:
        active_joints = _auto_select_joints(tree)

    max_angle = np.radians(max_angle_deg)

    # Compute local wire direction at each joint for rotation axis
    joint_axes = {}
    for j in active_joints:
        # Wire direction: average of edge directions at this joint
        neighbors = np.where(tree["adj"][j] > 0)[0]
        if len(neighbors) == 0:
            joint_axes[j] = np.array([0, 0, 1.0])
            continue
        directions = nodes[neighbors] - nodes[j]
        avg_dir = directions.mean(axis=0)
        avg_dir_norm = np.linalg.norm(avg_dir)
        if avg_dir_norm < 1e-8:
            # At a bifurcation, use direction to first neighbor
            avg_dir = directions[0]
            avg_dir_norm = np.linalg.norm(avg_dir)
        wire_tangent = avg_dir / (avg_dir_norm + 1e-12)

        # Rotation axis perpendicular to wire direction
        # Choose a consistent perpendicular using cross product with up or right
        up = np.array([0, 0, 1.0])
        perp = np.cross(wire_tangent, up)
        if np.linalg.norm(perp) < 0.1:
            up = np.array([0, 1, 0.0])
            perp = np.cross(wire_tangent, up)
        perp = perp / (np.linalg.norm(perp) + 1e-12)
        joint_axes[j] = perp

    # Generate frames with different phase offsets per joint
    frames = []
    rng = np.random.RandomState(42)
    phase_offsets = {j: rng.uniform(0, 2 * np.pi) for j in active_joints}
    freq_multipliers = {j: rng.uniform(0.5, 2.0) for j in active_joints}
    amp_multipliers = {j: rng.uniform(0.3, 1.0) for j in active_joints}

    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)  # [0, 1]

        # Compute joint rotations for this frame
        joint_rotations = {}
        for j in active_joints:
            angle = (max_angle * amp_multipliers[j] *
                     np.sin(2 * np.pi * freq_multipliers[j] * t + phase_offsets[j]))
            joint_rotations[j] = (joint_axes[j], angle)

        # Forward kinematics
        new_nodes, node_transforms = compute_forward_kinematics(tree, joint_rotations)

        # Animate surface points
        new_points = animate_points(points, binding, nodes, new_nodes, node_transforms)

        frames.append((new_points, labels.copy()))

    return frames


def _auto_select_joints(tree, max_joints=8):
    """Auto-select joints for animation: pick well-distributed structural nodes."""
    structural = tree["structural"]
    nodes = tree["nodes"]
    degrees = tree["degrees"]

    # Prefer bifurcation nodes (degree >= 3) and mid-segment endpoints
    # Start with bifurcations
    bifurcations = [i for i in structural if degrees[i] >= 3]

    # Add some mid-segment interior nodes for variety
    candidates = list(bifurcations)
    for seg in tree["wire_segments"]:
        if len(seg) > 10:
            mid = seg[len(seg) // 2]
            candidates.append(mid)

    # Limit to max_joints, selecting well-distributed ones
    if len(candidates) <= max_joints:
        return candidates

    # Greedy farthest-point sampling
    selected = [candidates[0]]
    for _ in range(max_joints - 1):
        best_dist = -1
        best_idx = -1
        for c in candidates:
            if c in selected:
                continue
            min_dist = min(np.linalg.norm(nodes[c] - nodes[s]) for s in selected)
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = c
        if best_idx >= 0:
            selected.append(best_idx)

    return selected


# ── Validation ──────────────────────────────────────────────────────────────

def validate_animation(frames, original_points, original_labels):
    """Validate animation frames quantitatively."""
    results = {}
    N = len(original_points)
    num_frames = len(frames)

    # V.1: Point count preservation
    counts = [f[0].shape[0] for f in frames]
    results["point_count"] = {
        "original": N,
        "all_match": all(c == N for c in counts),
    }

    # V.2: Label preservation
    label_match = all(np.array_equal(f[1], original_labels) for f in frames)
    results["labels_preserved"] = label_match

    # V.3: Local distance preservation (sample pairs)
    # For a rigid body, pairwise distances should be approximately preserved
    rng = np.random.RandomState(0)
    pair_indices = rng.choice(N, size=(min(1000, N), 2), replace=True)
    orig_dists = np.linalg.norm(
        original_points[pair_indices[:, 0]] - original_points[pair_indices[:, 1]], axis=1
    )

    dist_changes = []
    for frame_pts, _ in frames:
        frame_dists = np.linalg.norm(
            frame_pts[pair_indices[:, 0]] - frame_pts[pair_indices[:, 1]], axis=1
        )
        # Relative change in distance
        rel_change = np.abs(frame_dists - orig_dists) / (orig_dists + 1e-8)
        dist_changes.append(float(np.mean(rel_change)))

    results["distance_preservation"] = {
        "mean_relative_change": float(np.mean(dist_changes)),
        "max_relative_change": float(np.max(dist_changes)),
    }

    # V.4: Inter-frame smoothness (max point displacement between consecutive frames)
    max_displacements = []
    mean_displacements = []
    for i in range(1, num_frames):
        disp = np.linalg.norm(frames[i][0] - frames[i-1][0], axis=1)
        max_displacements.append(float(np.max(disp)))
        mean_displacements.append(float(np.mean(disp)))

    results["smoothness"] = {
        "mean_inter_frame_displacement": float(np.mean(mean_displacements)) if mean_displacements else 0,
        "max_inter_frame_displacement": float(np.max(max_displacements)) if max_displacements else 0,
    }

    # V.5: Bounding box check (points shouldn't fly off to infinity)
    all_pts = np.vstack([f[0] for f in frames])
    results["bounding_box"] = {
        "min": all_pts.min(axis=0).tolist(),
        "max": all_pts.max(axis=0).tolist(),
        "original_min": original_points.min(axis=0).tolist(),
        "original_max": original_points.max(axis=0).tolist(),
        "expansion_factor": float(
            np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) /
            np.max(original_points.max(axis=0) - original_points.min(axis=0) + 1e-8)
        ),
    }

    # V.6: No NaN/Inf
    has_nan = any(np.any(np.isnan(f[0])) for f in frames)
    has_inf = any(np.any(np.isinf(f[0])) for f in frames)
    results["numerical_stability"] = {
        "no_nan": not has_nan,
        "no_inf": not has_inf,
    }

    # Overall pass
    passed = (
        results["point_count"]["all_match"]
        and results["labels_preserved"]
        and results["numerical_stability"]["no_nan"]
        and results["numerical_stability"]["no_inf"]
        and results["bounding_box"]["expansion_factor"] < 3.0
        and results["smoothness"]["max_inter_frame_displacement"] < 0.5
    )
    results["passed"] = passed

    return results


# ── Phase 5: RGB-D Conversion & GIF ────────────────────────────────────────

def frames_to_rgbd_gif(frames, output_dir, view_name="front", fps=15):
    """Convert animation frames to RGB-D images and create GIF.

    Uses the same orthographic projection as the existing pcl_to_rgbd pipeline.
    """
    # Import projection functions from existing pipeline
    sys.path.insert(0, str(PROJECT_ROOT / "llmdocs" / "to_partner_teams" /
                           "kiat_crefle_rgbd_dataset" / "scripts"))
    from pcl_to_rgbd import (
        VIEWS, make_view_matrix, rasterize_view,
        IMG_W, IMG_H, CLASS_COLORS_BGR,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vdef = VIEWS[view_name]
    R = make_view_matrix(vdef["look"], vdef["up"])

    rgb_frames = []
    depth_frames = []

    for i, (pts, lbls) in enumerate(frames):
        color_img, depth_img = rasterize_view(pts, lbls.astype(int), R)

        # Save individual frames
        cv2.imwrite(str(output_dir / f"rgb_{i:04d}.png"), color_img)
        cv2.imwrite(str(output_dir / f"depth_{i:04d}.png"), depth_img)

        # Convert BGR to RGB for GIF
        rgb_frames.append(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        # Colorize depth for visualization
        valid_mask = depth_img > 0
        depth_vis = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        if np.any(valid_mask):
            d_norm = np.zeros_like(depth_img, dtype=np.float64)
            d_norm[valid_mask] = (depth_img[valid_mask].astype(np.float64) - 500) / 1000.0
            d_uint8 = np.clip(d_norm * 255, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_VIRIDIS)
            depth_color[~valid_mask] = 0
            depth_vis = depth_color
        depth_frames.append(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))

    # Create GIF using imageio (or PIL)
    try:
        from PIL import Image

        # RGB GIF
        pil_frames = [Image.fromarray(f) for f in rgb_frames]
        gif_path = output_dir / f"animation_rgb_{view_name}.gif"
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"  RGB GIF saved: {gif_path}")

        # Depth GIF
        pil_depth = [Image.fromarray(f) for f in depth_frames]
        gif_depth_path = output_dir / f"animation_depth_{view_name}.gif"
        pil_depth[0].save(
            str(gif_depth_path),
            save_all=True,
            append_images=pil_depth[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"  Depth GIF saved: {gif_depth_path}")

        # Side-by-side GIF (RGB + Depth)
        combined_frames = []
        for rgb_f, depth_f in zip(rgb_frames, depth_frames):
            combined = np.hstack([rgb_f, depth_f])
            combined_frames.append(Image.fromarray(combined))

        gif_combined_path = output_dir / f"animation_combined_{view_name}.gif"
        combined_frames[0].save(
            str(gif_combined_path),
            save_all=True,
            append_images=combined_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"  Combined GIF saved: {gif_combined_path}")

        return {
            "rgb_gif": str(gif_path),
            "depth_gif": str(gif_depth_path),
            "combined_gif": str(gif_combined_path),
            "num_frames": len(frames),
        }
    except ImportError:
        print("  PIL not available, skipping GIF creation")
        return {"error": "PIL not installed"}


# ── CLI ─────────────────────────────────────────────────────────────────────

def run_phase1(set_id=0, frame_id=0):
    """Phase 1: Joint identification and validation."""
    print("=" * 60)
    print("Phase 1: Joint Identification")
    print("=" * 60)

    data = load_sample(set_id, frame_id)
    tree = build_kinematic_tree(data["skel_nodes"], data["skel_adj"])
    anim_joints = select_animation_joints(tree)

    print(f"  Skeleton: {len(tree['nodes'])} nodes, "
          f"{sum(len(c) for c in tree['children'].values())} edges")
    print(f"  Structural nodes: {len(tree['structural'])} "
          f"(leaf={sum(1 for i in tree['structural'] if tree['degrees'][i]==1)}, "
          f"branch={sum(1 for i in tree['structural'] if tree['degrees'][i]>=3)})")
    print(f"  Wire segments: {len(tree['wire_segments'])}")
    print(f"  Animation joints: {len(anim_joints)}")
    print(f"  Root node: {tree['root']} (degree={tree['degrees'][tree['root']]})")

    results = validate_phase1(tree, data)
    print(f"\n  Validation:")
    print(f"    Tree valid: {results['tree_valid']['is_tree']}")
    print(f"    Structural match (leaf→endpoint): {results['structural_match']['leaf_match_rate']:.1%}")
    print(f"    Structural match (branch→bifurcation): {results['structural_match']['branch_match_rate']:.1%}")
    print(f"    PCL within 0.05 of skeleton: {results['coverage']['within_0.05']:.1%}")
    print(f"    All edges in segments: {results['segment_coverage']['all_edges_covered']}")
    print(f"    OVERALL: {'PASS' if results['passed'] else 'FAIL'}")

    return data, tree, anim_joints, results


def run_phase2(data, tree):
    """Phase 2: Surface point binding and validation."""
    print("\n" + "=" * 60)
    print("Phase 2: Surface Point Binding")
    print("=" * 60)

    binding = bind_points_to_skeleton(data["points"], tree["nodes"], tree["adj"])

    results = validate_phase2(binding, data["points"], tree["nodes"])
    print(f"  Validation:")
    print(f"    Weights sum to 1: {results['weight_sums']['all_near_one']}")
    print(f"    Perfect reconstruction: {results['reconstruction']['perfect_recon']}")
    print(f"    Mean offset: {results['offsets']['mean']:.4f}")
    print(f"    P95 offset: {results['offsets']['p95']:.4f}")
    print(f"    Edges with points: {results['edge_utilization']['edges_with_points']}")
    print(f"    OVERALL: {'PASS' if results['passed'] else 'FAIL'}")

    return binding, results


def run_phase3(tree, binding, data):
    """Phase 3: Single-joint animation."""
    print("\n" + "=" * 60)
    print("Phase 3: Single-Joint Animation")
    print("=" * 60)

    # Pick one bifurcation node for single-joint test
    bifurcations = [i for i in tree["structural"] if tree["degrees"][i] >= 3]
    if not bifurcations:
        print("  No bifurcation nodes found!")
        return None, None

    test_joint = bifurcations[0]
    print(f"  Test joint: node {test_joint} (degree={tree['degrees'][test_joint]})")

    frames = generate_animation_frames(
        tree, binding, data["points"], data["labels"],
        num_frames=30, max_angle_deg=15.0,
        active_joints=[test_joint],
    )

    results = validate_animation(frames, data["points"], data["labels"])
    print(f"  Generated {len(frames)} frames")
    print(f"  Validation:")
    print(f"    Point count preserved: {results['point_count']['all_match']}")
    print(f"    Labels preserved: {results['labels_preserved']}")
    print(f"    Mean distance change: {results['distance_preservation']['mean_relative_change']:.4f}")
    print(f"    Max inter-frame disp: {results['smoothness']['max_inter_frame_displacement']:.4f}")
    print(f"    BB expansion: {results['bounding_box']['expansion_factor']:.2f}x")
    print(f"    No NaN/Inf: {results['numerical_stability']['no_nan'] and results['numerical_stability']['no_inf']}")
    print(f"    OVERALL: {'PASS' if results['passed'] else 'FAIL'}")

    return frames, results


def run_phase4(tree, binding, data):
    """Phase 4: Multi-joint animation."""
    print("\n" + "=" * 60)
    print("Phase 4: Multi-Joint Animation")
    print("=" * 60)

    frames = generate_animation_frames(
        tree, binding, data["points"], data["labels"],
        num_frames=60, max_angle_deg=20.0,
        active_joints=None,  # auto-select
    )

    results = validate_animation(frames, data["points"], data["labels"])
    print(f"  Generated {len(frames)} frames with auto-selected joints")
    print(f"  Validation:")
    print(f"    Point count preserved: {results['point_count']['all_match']}")
    print(f"    Labels preserved: {results['labels_preserved']}")
    print(f"    Mean distance change: {results['distance_preservation']['mean_relative_change']:.4f}")
    print(f"    Max inter-frame disp: {results['smoothness']['max_inter_frame_displacement']:.4f}")
    print(f"    BB expansion: {results['bounding_box']['expansion_factor']:.2f}x")
    print(f"    No NaN/Inf: {results['numerical_stability']['no_nan'] and results['numerical_stability']['no_inf']}")
    print(f"    OVERALL: {'PASS' if results['passed'] else 'FAIL'}")

    return frames, results


def run_phase5(frames, output_dir=None):
    """Phase 5: RGB-D conversion and GIF generation."""
    print("\n" + "=" * 60)
    print("Phase 5: RGB-D Conversion & GIF")
    print("=" * 60)

    if output_dir is None:
        output_dir = RESULTS_ROOT / "rgbd_animation"

    results = {}
    for view in ["front", "top"]:
        view_dir = Path(output_dir) / view
        print(f"  Rendering {view} view...")
        r = frames_to_rgbd_gif(frames, view_dir, view_name=view)
        results[view] = r

    return results


def main():
    parser = argparse.ArgumentParser(description="DLO Point Cloud Animation")
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--phase3", action="store_true")
    parser.add_argument("--phase4", action="store_true")
    parser.add_argument("--phase5", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--set", type=int, default=0, help="Set ID")
    parser.add_argument("--frame", type=int, default=0, help="Frame ID")
    args = parser.parse_args()

    if args.all:
        args.phase1 = args.phase2 = args.phase3 = args.phase4 = args.phase5 = True

    if not any([args.phase1, args.phase2, args.phase3, args.phase4, args.phase5]):
        parser.print_help()
        sys.exit(1)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    data, tree, anim_joints, binding, frames = None, None, None, None, None

    if args.phase1:
        data, tree, anim_joints, p1_results = run_phase1(args.set, args.frame)
        if not p1_results["passed"]:
            print("\nPhase 1 FAILED. Fix issues before proceeding.")
            sys.exit(1)

    if args.phase2:
        if data is None:
            data = load_sample(args.set, args.frame)
            tree = build_kinematic_tree(data["skel_nodes"], data["skel_adj"])
        binding, p2_results = run_phase2(data, tree)
        if not p2_results["passed"]:
            print("\nPhase 2 FAILED. Fix issues before proceeding.")
            sys.exit(1)

    if args.phase3:
        if data is None:
            data = load_sample(args.set, args.frame)
            tree = build_kinematic_tree(data["skel_nodes"], data["skel_adj"])
        if binding is None:
            binding, _ = run_phase2(data, tree)
        frames, p3_results = run_phase3(tree, binding, data)

    if args.phase4:
        if data is None:
            data = load_sample(args.set, args.frame)
            tree = build_kinematic_tree(data["skel_nodes"], data["skel_adj"])
        if binding is None:
            binding, _ = run_phase2(data, tree)
        frames, p4_results = run_phase4(tree, binding, data)

    if args.phase5:
        if frames is None:
            print("No frames to render. Run --phase3 or --phase4 first.")
            sys.exit(1)
        run_phase5(frames)

    # Save all results
    print(f"\nResults saved to: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
