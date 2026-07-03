"""
post_process.py

This module provides post-processing utilities for kinematic preforming 
analyses.

Authors: Peter Hede Broberg, Christian Krogh, Bastian Schäfer, Jan Paul Wank, 
         Luise Kärger, Brian Lau Verndal Bak, Esben Lindgaard
Date: February 2026
"""
import numpy as np

def compute_e22_and_perp_strain(P, d):
    """
    Compute (1) E22 strain (transverse to fibre) and (2) perpendicular strain
    for each 4-node cell.

    Node order:
      node 1 = P[i][0], node 2 = P[i][1], node 3 = P[i][2], node 4 = P[i][3]

    Fibre direction edges:
      1->2 and 4->3

    Perpendicular strain:
      Average distance to BOTH fibre lines for symmetry.
    """
    def v(x):
        return np.asarray(x, dtype=float).reshape(3)

    def dist(a, b):
        return np.linalg.norm(v(b) - v(a))

    def point_line_distance(p, a, b):
        pa = v(p) - v(a)
        ba = v(b) - v(a)
        return np.linalg.norm(np.cross(pa, ba)) / np.linalg.norm(ba)

    n = len(P)
    e22 = np.zeros((n, 1), dtype=float)
    e_perp = np.zeros((n, 1), dtype=float)

    for i, cell in enumerate(P):
        n1, n2, n3, n4 = cell

        # E22: average of transverse edge strains based on distances (1-4) and (3-2)
        L14 = dist(n1, n4)
        L32 = dist(n3, n2)
        e22[i, 0] = 0.5 * ((L14 - d) / d + (L32 - d) / d)

        # Perpendicular: symmetric average distance to BOTH fibre lines (1-2) and (4-3)
        h4_to_12 = point_line_distance(n4, n1, n2)
        h3_to_12 = point_line_distance(n3, n1, n2)
        h1_to_43 = point_line_distance(n1, n4, n3)
        h2_to_43 = point_line_distance(n2, n4, n3)

        h_avg = 0.25 * (h4_to_12 + h3_to_12 + h1_to_43 + h2_to_43)
        e_perp[i, 0] = (h_avg - d) / d

    return e22, e_perp

