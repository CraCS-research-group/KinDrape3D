"""
kindrape_analysis.py

This module implements kinematic draping solvers for composite preforming.

The solvers compute the deformation of a single-layer preform over a 
double-curved tool surface using a grid-based discretization. The output 
includes the final node positions, cell corner points, and local shear angles.

Authors: Peter Hede Broberg, Christian Krogh, Bastian Schäfer, Jan Paul Wank, 
         Luise Kärger, Brian Lau Verndal Bak, Esben Lindgaard
"""
import numpy as np
from scipy.optimize import (fsolve, minimize)
from scipy.interpolate import splev

def kindrape_single_layer_combined(
        d, Grid, Org, Ang, OrgNode, F, method='standard', PreShear=0, 
        assumption = 'pure_shear', stabilizations = {}
        ):
    """
    Unified kinematic draping solver using either the standard or NR method.

    Args:
        d (float): Discretization spacing.
        Grid (list): Grid dimensions [rows, cols].
        Org (list): Starting point [x, y].
        Ang (float): Initial draping angle [deg].
        OrgNode (list): Grid node to place the origin.
        F (callable): Interpolator for mold surface.
        method (str): 'standard' or 'NR' for solver type.
        PreShear (float): Only used for NR method.

    Returns:
        tuple: (Shear, Node, P) where:
            - Shear is per-cell or per-edge shear
            - Node is 3D array of points
            - P is list of 4 corner tuples per cell
    """
    # Directions for expanding grid arms and quadrants
    Dir1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    Dir2 = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
    
    # Initialize data structures
    Node = np.full((*Grid, 3), np.nan)
    P = [list(((np.nan, np.nan, np.nan),) * 4)] * (Grid[0] - 1) * (Grid[1] - 1)
    Shear = np.full(((Grid[0] - 1) * (Grid[1] - 1), 4), np.nan)
    
    # STEP 1: Place org. node (1) and node (2) defined by ini. drape angle
    # Get indices for cell, place 1st node and solve for 2nd node
    Idx = get_cell_idx(OrgNode[0], OrgNode[1], Dir1, Dir2, Grid, 0)[0]
    Node[tuple(Idx[:, 0])] = [Org[0], Org[1], F(Org[0], Org[1])]

    if method == 'standard':
        a_sol = fsolve(DistFun, d, args=(Node[tuple(Idx[:, 0:2])], F, d, Ang))
        Node[tuple(Idx[:, 0:2])] = CellVertCoor(a_sol,
                                                Node[tuple(Idx[:, 0:2])], 
                                                F, Ang)
    else:
        Node[tuple(Idx)] = MoldCircIntersec(Node[tuple(Idx)], F, d, 
                                            Ang + 90, 1)

    # STEP 2: Place generator cells (initial cells) while minimizing shear
    GenStart = OrgNode + np.array([[0, 0], [1, 1], [0, 1], [0, 0]])
    nGen = np.hstack([Grid[0] - OrgNode[0] - 1, 
                      Grid[1] - OrgNode[1] - 2, 
                      OrgNode]) # number of cells per arm
    Opt = {'maxiter': 100, 'ftol': 5e-9, 'iprint': 0, 'disp': True}
    
    for i in range(4): # For each arm
        CellAng = Ang + i * 90 + PreShear * (1 + (-1)**i) / 2
        # Initial guess
        a_0 = 0.75 * d * np.tile(CosSin(CellAng), (1, 2)).flatten() 

        for k in (GenStart[i, :] 
                  + np.arange(nGen[i]).reshape(-1, 1) * Dir1[i, :]):
            # Define idx
            Idx, CellNo = get_cell_idx(k[0], k[1], Dir1, Dir2, Grid, i)

            if method == 'standard':
                # Define solver input, call optimizer, assign solution
                BndBox = 0.5 * np.array([[-1]*4, [1]*4]) 
                Bnd = tuple(map(tuple, 
                                (a_0 + np.linalg.norm(a_0[0:3]) * BndBox).T))
                Arg = (Node[tuple(Idx)], F, d)
                Constr = {'type': 'eq', 'fun': DistFun, 'args': Arg}                         
                res = minimize(ObjFun, a_0, args=Arg, bounds=Bnd, 
                               method='SLSQP', constraints=Constr, options=Opt)
                Node[tuple(Idx)] = CellVertCoor(res.x, Node[tuple(Idx)], F)
                a_0 = res.x # use as next initial guess
            else:
                # Solve for Vert #2+4 using NR, upd. CellAng_0
                Node[tuple(Idx)], CellAng = NRSol(Node[tuple(Idx)], CellAng, F, 
                                                  d, PreShear, i)

            # Put current cell coord. and shear in P and Shear and update a_0
            P_c = list(map(tuple, Node[tuple(Idx)]))
            P[CellNo] = [P_c[(i + j) % 4] for j in range(4)]
            shear_val = compute_cell_shear(Node[tuple(Idx)], method, i)
            Shear[CellNo] = (shear_val if method == 'standard' 
                                 else np.mean(shear_val))

    # STEP 3: Place remaining, constrained cells in 4 quadrants between arms
    ConStart = OrgNode + np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
    nCon = (nGen[[0, 1, 2, 1, 2, 3, 0, 3]] 
                - [1, 0, 0, 0, 0, 0, 1, 0]).reshape(4, 2)

    for i in range(4): # For each quadrant
        # Fibre traking algorithm for simple shear solvers
        if method == 'NR' and (assumption == 'simple_shear' 
                                   or assumption == 'enhanced_simple_shear'):
            h_list = (ConStart[i, 0] 
                          + np.arange(nCon[i, 0]) * (Dir1[i, 0] + Dir2[i, 0]))
            w_list = (ConStart[i, 1] 
                          + np.arange(nCon[i, 1]) * (Dir1[i, 1] + Dir2[i, 1]))
            if len(h_list) == 0 or len(w_list) == 0:
                continue
            fibers, L_0 = fiber_tracking(h_list, w_list, Dir1, Dir2, Grid, i, 
                                         Node, d, F, assumption, 
                                         stabilizations)
            
        # Placing of the remaining constrained cells
        h_idx = 0
        for h in (ConStart[i, 0] 
                      + np.arange(nCon[i, 0]) * (Dir1[i, 0] + Dir2[i, 0])):
            w_idx = 0
            for w in (ConStart[i, 1] 
                          + np.arange(nCon[i, 1]) * (Dir1[i, 1] + Dir2[i, 1])):
                # Get cell idx and no. 
                Idx, CellNo = get_cell_idx(h, w, Dir1, Dir2, Grid, i)

                if method == 'standard' and assumption == 'pure_shear':
                    # Define solver input, call fsolve, assign solution
                    a_0 = (Node[tuple(Idx[:, 3])][0:2] 
                               - Node[tuple(Idx[:, 0])][0:2])
                    a_sol = fsolve(DistFun, a_0, args=(Node[tuple(Idx)], F, d),
                                   factor=1, diag=(d, d))
                    Node[tuple(Idx)] = CellVertCoor(a_sol, Node[tuple(Idx)], F)
                elif (method == 'NR' 
                          and (assumption == 'simple_shear' 
                                   or assumption == 'enhanced_simple_shear')):
                    # Call find_nodes_simple_shear to find Vert #3
                    new_point = find_nodes_simple_shear(fibers[h_idx + 1], 
                                                        (w_idx + 1) * d)
                    vert = Node[tuple(Idx)]
                    vert[2,:] = new_point
                    Node[tuple(Idx)] = vert
                else:
                    # Call MoldCircIntersec to get Vert #3
                    Node[tuple(Idx)] = MoldCircIntersec(Node[tuple(Idx)], F, d, 
                                                        [], 2)
                
                # Put current cell coord. and shear in P and Shear
                P_c = list(map(tuple, Node[tuple(Idx)]))
                P[CellNo] = [P_c[(i + j) % 4] for j in range(4)]
                shear_val = compute_cell_shear(Node[tuple(Idx)], method, i)
                Shear[CellNo] = (shear_val if method == 'standard' 
                                     else np.mean(shear_val))
                w_idx += 1
            h_idx += 1

    return Shear, Node, P

def fiber_tracking(h_list, w_list, Dir1, Dir2, Grid, i, Node, d, F, assumption, 
                   stabilizations):
    fibers = []
    # Define fiber 1 (found from the generators)
    Idx, CellNo = get_cell_idx(h_list[0], w_list[0], Dir1, Dir2, Grid, i)
    fibers_1 = [Node[tuple(Idx)][0]]
    for w in w_list:
        Idx, CellNo = get_cell_idx(h_list[0], w, Dir1, Dir2, Grid, i)
        if (i == 0 or i == 2):
            fibers_1.append(Node[tuple(Idx)][1])
        else:
            fibers_1.append(Node[tuple(Idx)][3])
    fibers.append(fibers_1)
    
    L_0 = len(w_list)*d # Length of the fibre
    
    # Compute the rest of the fibers
    h_idx = 0
    for h in h_list: # For each fibre
        fiber_new = []
        w_idx = 0
        dir_idx = 0
        L_fiber = 0
        # Find first point 
        Idx, CellNo = get_cell_idx(h, w_list[0], Dir1, Dir2, Grid, i)
        if (i == 0 or i == 2): # Depend on quadrant
            fiber_new.append(Node[tuple(Idx)][3])
        else:
            fiber_new.append(Node[tuple(Idx)][1])
        # Find new points until the length is obtained
        d_updated = d ## Start distance for the damping
        while L_fiber < L_0:
            # Call the transverse deformation update to output the updated d 
            if assumption == 'enhanced_simple_shear':
                d_previous = d_updated
                d_updated = compute_transverse_deformation(
                    fibers[h_idx], L_fiber, L_0, d, fiber_new[-1], 
                    stabilizations['enhanced_simple_shear_parameters'], 
                    stabilizations['enhanced_simple_shear_relation_type']
                    )
                d_updated = viscous_damping(
                    d_updated, d_previous, 
                    stabilizations['enhanced_simple_shear_damping']
                    )
            if w_idx + 1  < len(fibers[h_idx]) - 1: 
                # Find the next fibre point from cylinder-cylinder intersection 
                # if there are at least two fibre points on the previous fibre
                vert_a = np.array(
                    [fibers[h_idx][w_idx], fibers[h_idx][w_idx+1], 
                     np.zeros(3)*np.nan, fiber_new[w_idx-dir_idx]]
                    )
                vert_b = np.array(
                    [fibers[h_idx][w_idx], fiber_new[w_idx-dir_idx], 
                     np.zeros(3)*np.nan, fibers[h_idx][w_idx+1]]
                    )
                vert = vert_a if (i == 0 or i == 2) else vert_b
                vert_1 = np.array(
                    [fibers[h_idx][w_idx+2], fibers[h_idx][w_idx+2], 
                     fibers[h_idx][w_idx+2], fibers[h_idx][w_idx+2]]
                    )
                if not stabilizations['average_fiber_length']:
                    fiber_new.append(
                        simple_shear_function(vert,vert_1, F, d_updated, 2, 
                                              i, False, fibers[h_idx-1])[2]
                        )
                else: 
                    fiber_new.append(
                        simple_shear_function(vert,vert_1, F, d_updated, 2, 
                                              i, True, fibers[h_idx-1])[2]
                        )
            else:
                vert_a = np.array(
                    [fibers[h_idx][len(fibers[h_idx]) - 2], 
                     fibers[h_idx][len(fibers[h_idx]) - 1], 
                     np.zeros(3)*np.nan, 
                     fiber_new[w_idx-dir_idx]]
                    )
                vert_b = np.array(
                    [fibers[h_idx][len(fibers[h_idx]) - 2], 
                     fiber_new[w_idx-dir_idx], 
                     np.zeros(3)*np.nan, 
                     fibers[h_idx][len(fibers[h_idx]) - 1]]
                    )
                vert = vert_a  if (i == 0 or i == 2) else vert_b
                fiber_new.append(
                    SphereCylinderIntersec(vert, F, d_updated, 2, i, R=d)[2]
                    ) 
            
            
            # Check if the points are turning backwards, and adjust accordingly
            if stabilizations['remove_spurious_points']:
                if w_idx - dir_idx > 0 and dir_idx < 50: # Dir idx max added
                    vector_1 = (fiber_new[w_idx-dir_idx] 
                                    - fiber_new[w_idx-1-dir_idx])
                    vector_2 = (fiber_new[w_idx+1-dir_idx] 
                                    - fiber_new[w_idx-dir_idx])
                    check_dir = compute_angle_between_vectors(vector_1, 
                                                              vector_2)
                    if check_dir > np.deg2rad(90): 
#                        print('CRemoval of spurious points has been called!!')
                        # Delete the appropriate length
                        L_delete= np.linalg.norm(
                            fiber_new[w_idx-dir_idx]-fiber_new[w_idx-1-dir_idx]
                            )
                        L_fiber = L_fiber - L_delete
                        del fiber_new[w_idx-dir_idx]
                        # Increase number corresponding to times direction
                        # has changed
                        dir_idx += 1 
                        
                        
            # Find the length of the fibre section and update L_fiber
            # Update w_idx
            L_section = np.linalg.norm(
                fiber_new[w_idx+1-dir_idx] - fiber_new[w_idx-dir_idx]
                ) # Insert fibre length
            L_fiber = L_fiber + L_section
            w_idx += 1
        h_idx += 1
        
        # Remove the points that causes too high curvature
        if stabilizations['curvature_based_point_removal']:
            fiber_new = stabilise_fibre_by_max_curvature(
                fiber_new, stabilizations['curvature_threshold'], 
                irregularity = stabilizations['use_irregularity_index']
                ) 
        fibers.append(fiber_new)
    #kd3d.utility.plot_fibres(fibers)
    return fibers, L_0


def point_on_fibre(P, F, tol=1e-6): 
    """
    Auxillary function to check if a new point is placed on a previous fibre.
    """
    P = np.asarray(P)
    for A, B in zip(F[:-1], F[1:]):
        A, B = np.asarray(A), np.asarray(B)
        AB = B - A
        AP = P - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        if 0.0 <= t <= 1.0:
            dist = np.linalg.norm(AP - t * AB)
            if dist < tol:
                return True
    return False

def viscous_damping(d_new, d_prev, damping_coeff, dt = 1):
    d_damped = (d_new * dt / (damping_coeff + dt) 
                    + d_prev * damping_coeff / (damping_coeff + dt))
    return d_damped

def compute_transverse_deformation(fiber, L_fiber, L_0, d, fibre_start, 
                                   parameters, relation_type):
    ## Step 1: Find two points at L_fiber and L_fiber+d
    point1, point2 = find_shear_nodes(fiber, L_fiber)
    point3 = fibre_start
    
    ## Step 2: Compute the shear angle or angle between the fibers
    vector1 = point2 - point1
    vector2 = point3 - point1
    angle = compute_angle_between_vectors(vector1, vector2)
    shear_angle =  np.pi/2 - angle 
  
    ## Step 3: Compute the updated distance as a function of the shear
    if relation_type == 'phenomenological':
        d_updated = d - np.tanh(parameters[0]*shear_angle)*parameters[1]*d
    elif relation_type == 'tricot' or relation_type == 'tricot_bounded':
        angle = parameters[0]
       # a = 10
        tricot_angle = np.deg2rad(angle)
        d_0 = parameters[1]*d
        theoretical = (d*np.sin(shear_angle) 
                           * (np.sqrt(1 + (1/np.tan(tricot_angle)**2)
                                          * (np.cos(shear_angle))**2) 
                                  - 1 / np.tan(tricot_angle)
                                      * np.cos(shear_angle)))
        d_updated = max(theoretical, d_0) 
        if relation_type == 'tricot_bounded':
            if shear_angle < 0:
                d_updated = d
    elif relation_type == 'simple_shear':
        d_updated = d
    elif relation_type == 'spline':
        ##### ALTERNATIVE step with the new spline ##############
        gamma_lock = np.deg2rad(90)  # lock angle (radians)
        k = 3                   # cubic spline
        n_ctrl = len(parameters)# number of coefficients

        t = make_open_uniform_knots(n_ctrl, k)
        # --------------------------------------------------------------------
        # 3. Evaluate over range of shear angles
        # --------------------------------------------------------------------
        d_updated = d_from_spline(shear_angle, d, parameters, t, k, gamma_lock)
        #####################################
    else: 
        raise ValueError("Unknown shear-spacing relation type inputted.")
    return d_updated

def make_open_uniform_knots(n_ctrl: int, k: int):
    """
    Open, clamped knot vector on the full signed domain s ∈ [-1, 1].
    End knots have multiplicity k+1 to clamp the spline.
    """
    if n_ctrl <= k:
        raise ValueError("Need n_ctrl > k for a valid spline.")
    n_int = n_ctrl - k - 1                     # number of internal knots
    internal = np.linspace(-1.0, 1.0, n_int + 2)[1:-1] if n_int > 0 else []
    return np.r_[np.full(k+1, -1.0), internal, np.full(k+1,  1.0)]

def d_from_spline(gamma, d0, c, t, k, gamma_lock=np.deg2rad(45.0), 
                  normalize_at_zero=False):
    """
    Evaluate d(gamma) on signed normalized shear 
    s = gamma / gamma_lock ∈ [-1, 1].
    If normalize_at_zero=True, we rescale so that d(0) = d0 
    regardless of coefficients.
    """
    s = np.clip(gamma / gamma_lock, -1.0, 1.0)  # <-- signed, allows asymmetry
    y = np.asarray(splev(s, (t, c, k), ext=2))

    if normalize_at_zero:
        y0 = float(splev(0.0, (t, c, k), ext=2))
        if abs(y0) > 1e-12:
            y = y / y0  # ensures d(0) = d0
    return d0 * y

def find_shear_nodes(fiber, L_node):
    # Find length target
    if L_node == 0:
        return fiber[0], fiber[1]
    
    # Loop over fibers to find 
    L_acc = 0
    L_idx = 0
    while L_acc < L_node: # The solution is between L_idx-L_idx
    
        ################ Simple bounds check to place missing nodes ###########
        if L_idx + 1 >= len(fiber):
#            print('An auxillary point outside the fibre has been placed in '
#                  'shear tracking')
            A, B = fiber[-2], fiber[-1]
            direction = (B - A) / np.linalg.norm(B - A)
            return B + direction * (L_node - L_acc), fiber[-1]
        #######################################################################
    
        L_int = np.linalg.norm(fiber[L_idx+1]-fiber[L_idx])
        L_acc += L_int
        L_idx += 1
        
    ## Todo: if L_idx exceeds the length, the loop should stop.
    # Find excessive length
    L_excess = L_node - (L_acc-L_int) 
    L_ratio = L_excess / L_int 
    
    # Interpolate between the nearest points
    new_point = fiber[L_idx - 1] * (1 - L_ratio) + fiber[L_idx] * L_ratio
    if L_ratio < 0.9999:
        shear_point = fiber[L_idx]
    else:
        shear_point = fiber[L_idx-1]
#        print('Reverse shearing')

    return new_point, shear_point

def find_nodes_simple_shear(fiber, L_node):
    # Find length target
    if L_node == 0:
        return fiber[0]
    
    # Loop over fibers to find 
    L_acc = 0
    L_idx = 0
    while L_acc < L_node: # The solution is between L_idx-L_idx
    
        ################ Simple bounds check to place missing nodes ###########
        if L_idx + 1 >= len(fiber):
#            print('An auxillary point outside the fibre has been placed')
            A, B = fiber[-2], fiber[-1]
            direction = (B - A) / np.linalg.norm(B - A)
            return B + direction * (L_node - L_acc)
        #######################################################################
    
        L_int = np.linalg.norm(fiber[L_idx+1]-fiber[L_idx])
        L_acc += L_int
        L_idx += 1
        
    ## Todo: if L_idx exceeds the length, the loop should stop.
    
    # Find excessive length
    L_excess = L_node - (L_acc-L_int) 
    L_ratio = L_excess / L_int 
    
    # Interpolate between the nearest points
    new_point = fiber[L_idx - 1] * (1 - L_ratio) + fiber[L_idx] * L_ratio
    return new_point

def fibre_curvature_circumcircle(points, eps=1e-12, return_irregularity=False, 
                                 ratio_cap=None):
    """
    Compute discrete curvature (magnitude) at each interior point of a fibre
    using the circumcircle method:
        κ_i = 4A / (a b c) 
            = 2 * ||(P_i - P_{i-1}) × (P_{i+1} - P_i)|| / (a b c)

    Parameters
    ----------
    points : (N, 3) array_like or list of array_like
        Ordered 3D coordinates of the fibre. Endpoints are treated as pinned.
    eps : float, optional
        Small positive threshold to avoid division by ~0.

    Returns
    -------
    kappa : (N,) ndarray
        Curvature magnitude at each point (1/length units). Endpoints are 0.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be an array of shape (N, 3)")
    N = P.shape[0]
    if N < 3:
        # Not enough points to define curvature
        return np.zeros(N, dtype=float)

    # Vectors for interior triples
    v1 = P[1:-1] - P[:-2]          # (N-2, 3) = P_i - P_{i-1}
    v2 = P[2:]   - P[1:-1]         # (N-2, 3) = P_{i+1} - P_i

    # Side lengths a, b, c
    a = np.linalg.norm(v1, axis=1)                     # ||P_i - P_{i-1}||
    b = np.linalg.norm(v2, axis=1)                     # ||P_{i+1} - P_i||
    c = np.linalg.norm(P[2:] - P[:-2], axis=1)         # ||P_{i+1} - P_{i-1}||

    # Twice the triangle area = ||v1 × v2||
    cross_v = np.cross(v1, v2)
    A2 = np.linalg.norm(cross_v, axis=1)

    # κ = 2 * A2 / (a b c), with guards for near-degenerate triples
    denom = a * b * c
    valid = (A2 > eps) & (denom > eps)
    kappa_int = np.zeros(N - 2, dtype=float)
    kappa_int[valid] = (2.0 * A2[valid]) / denom[valid]

    # Assemble full array with zeros at endpoints
    kappa = np.zeros(N, dtype=float)
    kappa[1:-1] = kappa_int
    
    if not return_irregularity:
        return kappa

    # Irregularity ratio: max(a,b) / max(min(a,b), eps)
    min_ab = np.maximum(np.minimum(a, b), eps)
    ratio = np.maximum(a, b) / min_ab
    if ratio_cap is not None:
        ratio = np.minimum(ratio, ratio_cap)
    irr_int = kappa_int * ratio
    irregularity = np.zeros(N, dtype=float)
    irregularity[1:-1] = irr_int

    return irregularity

def stabilise_fibre_by_max_curvature(fibre_pts, kappa_max, eps=1e-12, 
                                     irregularity=False): 
    """
    Stabilise a fibre (list of (3,) arrays) by repeatedly removing the interior 
    point with the highest circumcircle curvature if that removal lowers the 
    global maximum curvature. Stops when the max curvature <= kappa_max.

    Requires: fibre_curvature_circumcircle(points, eps)

    Parameters
    ----------
    fibre_pts : list[np.ndarray]  # each element shaped (3,)
    kappa_max : float             # threshold (1/length)
    eps : float                   # small guard for curvature computation

    Returns
    -------
    list[np.ndarray]  # same format as input (each (3,))
    """
    # Convert to (N,3) array
    P = np.asarray(fibre_pts, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("fibre_pts must be a list/array of shape (N, 3) with "
                         " 1D points.")

    if P.shape[0] < 3:
        return [row.copy() for row in P]

    def kappa_arr(Q):
        return fibre_curvature_circumcircle(Q, eps=eps, 
                                            return_irregularity = irregularity)

    while True:
        kappa = kappa_arr(P)
        if P.shape[0] < 3:
            break

        # interior max (endpoints have zero curvature in this scheme)
        k_int = kappa[1:-1]
        if k_int.size == 0:
            break

        old_max = float(k_int.max())
        if old_max <= kappa_max:
            break

        # index of worst interior point in full indexing
        i_star = 1 + int(np.argmax(k_int))

        # Try deleting the worst point
        P_try = np.delete(P, i_star, axis=0)
        if P_try.shape[0] < 3:
            # cannot evaluate curvature meaningfully anymore
            break

        new_kappa = kappa_arr(P_try)
        new_int = new_kappa[1:-1] if P_try.shape[0] > 2 else np.array([])
        new_max = float(new_int.max()) if new_int.size else 0.0

        if new_max < old_max:
            P = P_try  # accept deletion and continue
#            print('Pruning done')
        else:
            # no improvement → stop
            break

    # Return in original format: list of (3,) arrays
    return [row.copy() for row in P]

def simple_shear_function(Vert, Vert_1, F, d, UnknownVertIdx, qd, 
                          average_fiber_length = True, previous_fibre=[]):
    
    # Define the points for the analysis
    P1 = Vert[[0], :][0]
    P2 = Vert[[1], :][0] if (qd == 0 or qd == 2) else Vert[[3], :][0]
    P3 = Vert_1[[1], :][0] if (qd == 0 or qd == 2) else Vert_1[[3], :][0] 
    P4 = Vert[[3], :][0] if (qd == 0 or qd == 2) else Vert[[1], :][0]
    
    
    #### Compute the cut-off angle for sphere-cylnder intersection
    angle = compute_angle_between_vectors(P3-P2, P1-P2) 
    cut_off_angle = np.radians(5)
    if (angle < cut_off_angle 
            or angle > (np.pi - cut_off_angle) 
                or np.isnan(angle)):
        if average_fiber_length:
            # Inserted to stabilise results 
            R = (np.linalg.norm(P1-P2) + np.linalg.norm(P3-P2)) / (2) 
        else:
            R = d
        Vert = SphereCylinderIntersec(Vert, F, d, UnknownVertIdx, qd, R)
        return Vert
    
    IntersecPt = cylinder_cylinder_intersection(P1, P2, P3, P4, F, d, 
                                                cut_off_angle, previous_fibre)
    
    # Store the IntersecPt in Vert at the row specified by UnknownVertIdx        
    Vert[UnknownVertIdx,:] = IntersecPt
    return Vert

def cylinder_cylinder_intersection(P1, P2, P3, P4, F, d, cut_off_angle, 
                                   previous_fibre):
    ## Establish local orthonormal basis
    # 1: Place z along 1-2
    cylinder_axis = P2 - P1
    L_axis = np.linalg.norm(cylinder_axis)
    if L_axis < 1e-12:
        raise ValueError("Cylinder axis is degenerate: P1 and P2 coincide.")
    z_hat = cylinder_axis / L_axis
    
    # 2: Place y normal to the plane defined by 1-2-3
    y_hat_temp = np.cross(P3-P2, P1-P2)
    
    # Ensures that y hat points in the direction of point 4. 
    # This is for easier finding the roots later on. 
    if np.dot(P4-P2, y_hat_temp) < 0:  
        y_hat_temp = y_hat_temp * (-1)
    
    y_hat = (y_hat_temp)/ np.linalg.norm(y_hat_temp)
    
    # 3: Place x according to right hand rule
    x_hat = np.cross(y_hat,z_hat) / np.linalg.norm(np.cross(y_hat,z_hat))
    
    ## Find theta between the two cylinders
    theta = np.arccos(np.dot(P2-P1,P3-P2) 
                          / (np.linalg.norm(P2-P1) 
                              * np.linalg.norm(P3-P2)))
    
    if np.dot(P3-P2, x_hat) < 0:  # Ensures the correct sign of theta
        theta = - theta
        
    ## Search for the intersection between first curve
    
    r_cylinders = d
    
    # Bisection
    # Find the intersection between the curve and the surface using bisection 
    IntersecPt = np.array([np.nan, np.nan, np.nan])

    phi = np.array([- 2/4 * np.pi, 3/4 * np.pi]) # - 1/4 * np.pi, 3/4 * np.pi
    
    x_local = r_cylinders * np.cos(phi[0])
    y_local = r_cylinders * np.sin(phi[0])
    z_local = (1 / np.tan(theta) - 1 / np.sin(theta)) * x_local #*(-1) #removed pi
    CurveCoor = P2 + x_local*x_hat + y_local*y_hat + z_local*z_hat
    FunVal_beginning = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
    
    x_local = r_cylinders * np.cos(phi[1])
    y_local = r_cylinders * np.sin(phi[1])
    z_local = (1 / np.tan(theta) - 1 / np.sin(theta)) * x_local #*(-1) #removed pi
    CurveCoor = P2 + x_local*x_hat + y_local*y_hat + z_local*z_hat
    FunVal_ending = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
    
    increasing = FunVal_ending >  FunVal_beginning
    
    for i in range(100): # Max iter
        # Compute middle point
        phi_mid = np.sum(phi)/2
        # Curve pt. in 3D based on center, radius, 2 perp. vectors and angle
        x_local = r_cylinders * np.cos(phi_mid)
        y_local = r_cylinders * np.sin(phi_mid)
        z_local = (1 / np.tan(theta) - 1 / np.sin(theta)) * x_local #*(-1) #removed pi
        CurveCoor = P2 + x_local*x_hat + y_local*y_hat + z_local*z_hat
        FunVal_mid = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
        # Stop or adjust interval based on function value at midpoint
        if np.abs(FunVal_mid) < 5e-6: # Stopping tolerance
            IntersecPt = np.hstack((CurveCoor[0:2].T,F(CurveCoor[0],
                                                       CurveCoor[1])))
            break
        elif FunVal_mid < 0 and increasing: # Changed sign on this one
            phi[0] = phi_mid
        elif FunVal_mid > 0 and not increasing: # Changed sign on this one
            phi[0] = phi_mid
        else:
            phi[1] = phi_mid
        
        if i == 99: # Last iteration
            IntersecPt = np.hstack((CurveCoor[0:2].T,F(CurveCoor[0],
                                                       CurveCoor[1])))
   
    check_angle = compute_angle_between_vectors(P2-P1, IntersecPt-P4) 
    if (np.pi - cut_off_angle >  check_angle > cut_off_angle 
            or abs(FunVal_mid) > 1e-3):
        # Bisection
        # Find the intersection between the curve and the surface using
        IntersecPt_alt = np.array([np.nan, np.nan, np.nan])

        phi = np.array([1/4 * np.pi, 6/4 * np.pi]) # # 1/4 * np.pi, 5/4 * np.pi
        
        x_local = r_cylinders * np.cos(phi[0])
        y_local = r_cylinders * np.sin(phi[0])
        z_local = (1 / np.tan(theta) - 1 / np.sin(theta)) * x_local #*(-1) #removed pi
        CurveCoor = P2 + x_local*x_hat + y_local*y_hat + z_local*z_hat
        FunVal_beginning = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
        
        x_local = r_cylinders * np.cos(phi[1])
        y_local = r_cylinders * np.sin(phi[1])
        z_local = (1 / np.tan(theta) - 1 / np.sin(theta)) * x_local #*(-1) # removed pi
        CurveCoor = P2 + x_local*x_hat + y_local*y_hat + z_local*z_hat
        FunVal_ending = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
        
        increasing = FunVal_ending >  FunVal_beginning
        
        for i in range(100): # Max iter
            # Compute middle point
            phi_mid = np.sum(phi)/2
            # Curve pt. in 3D based on center, radius, 2 perp. vectors and ang
            x_local = r_cylinders * np.cos(phi_mid)
            y_local = r_cylinders * np.sin(phi_mid)
            z_local = (1 / np.tan(theta) - 1 / np.sin(theta)) * x_local #*(-1) # removed pi
            CurveCoor = P2 + x_local*x_hat + y_local*y_hat + z_local*z_hat
            FunVal_mid_alt = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
            # Stop or adjust interval based on function value at midpoint
            if np.abs(FunVal_mid_alt) < 5e-6: # Stopping tolerance
                IntersecPt_alt = np.hstack((CurveCoor[0:2].T,F(CurveCoor[0],
                                                               CurveCoor[1])))
                break
            elif FunVal_mid_alt < 0 and increasing: # Changed sign on this one
                phi[0] = phi_mid
            elif FunVal_mid_alt > 0 and not increasing: # Changed sign on this one
                phi[0] = phi_mid
            else:
                phi[1] = phi_mid
                
            if i == 99: # Last iteration
                IntersecPt_alt = np.hstack((CurveCoor[0:2].T,F(CurveCoor[0],
                                                               CurveCoor[1])))
        
        ########### Test to see if the bew point is shifting fibres
        if point_on_fibre(IntersecPt, previous_fibre, tol=d):
            IntersecPt = IntersecPt_alt 
#            print('Fibre crossing detected')
        elif point_on_fibre(IntersecPt_alt, previous_fibre, tol=d):
            IntersecPt_alt = IntersecPt    
#            print('Fibre crossing detected')
            
        if (point_on_fibre(IntersecPt, previous_fibre, tol=d) 
                and point_on_fibre(IntersecPt_alt, previous_fibre, tol=d)):
            print('Both points are on the wrong fibre!!!!!!!!!!!')
        #################
        
        # Test to see if the new point is actually better
        if abs(FunVal_mid) <  5e-6 and abs(FunVal_mid_alt) <  5e-6:
            check_angle_alt = compute_angle_between_vectors(P2-P1,
                                                            IntersecPt_alt-P4)
            angle = min(np.pi - check_angle, check_angle)
            angle_alt = min(np.pi - check_angle_alt, check_angle_alt)
            if angle >  angle_alt:
                IntersecPt = IntersecPt_alt 
        elif abs(FunVal_mid) <  5e-6 or abs(FunVal_mid_alt) <  5e-6:
            if abs(FunVal_mid_alt) <  5e-6:
                IntersecPt = IntersecPt_alt 
        else:
            raise ValueError(
                "No root found in the cylinder-cylinder bisection."
                )
         
    return IntersecPt

def compute_angle_between_vectors(vector1, vector2):
    length = (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if length < 1e-16:
        raise ValueError("Cannot divide by zero.")
    value = np.dot(vector1,vector2) / length
    if value > 1:
        angle = 0
    elif value < -1:
        angle = np.pi
    else:
        angle = np.arccos(value)
    return angle


def get_cell_idx(Row, Col, Dir1, Dir2, Grid, No):
    """
    Returns index array and cell number for a given vertex 1.

    Supports both standard and NR solvers.
    """
    Rows = Row + np.array([0, Dir2[No, 0], Dir1[No, 0] 
                               + Dir2[No, 0], Dir1[No, 0]])
    Cols = Col + np.array([0, Dir2[No, 1], Dir1[No, 1] 
                               + Dir2[No, 1], Dir1[No, 1]])
    CellNo = Rows[No] + Cols[No] * (Grid[0] - 1)
    return np.vstack([Rows, Cols]), CellNo

def compute_cell_shear(Vert, method, quadrant_index=None):
    """
    Computes 4 shear values per cell (standard) or signed shear (NR).

    Args:
        Vert (ndarray): 4x3 vertex coordinates.
        method (str): 'standard' or 'NR'.
        quadrant_index (int): Needed for NR to resolve sign convention.

    Returns:
        ndarray: shear angles per cell edge [deg]
    """
    u = Vert[[1, 2, 3, 0]] - Vert
    v = Vert[[3, 0, 1, 2]] - Vert
    angle = np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.sum(u * v, 
                                                                      axis=1))

    if method == 'standard':
        return np.abs(angle * 180 / np.pi - 90)
    else:
        shear = angle - np.pi / 2
        sign = np.array([-1, 1, -1, 1]) * (-1)**quadrant_index
        return shear * sign * 180 / np.pi

def DistFun(a, Vert, F, d, Ang=0.0):
    """
    Distance constraint function used in optimization and fsolve.

    Computes the difference between current edge lengths and desired spacing, 
    d, for use in enforcing constant spacing in cells.

    Parameters:
        a (ndarray): Design variable controlling node positions.
        Vert (ndarray): (4, 3) array of current vertex coordinates.
        F (callable): Surface interpolator (z = F(x, y)).
        d (float): Target edge length.
        Ang (float): Draping angle in degrees (used if a.size == 1).

    Returns:
        ndarray: Difference between actual and target edge lengths.
    """
    Vert = CellVertCoor(a, Vert, F, Ang)
    
    if a.size == 4:
        EdgeLength = np.linalg.norm(
            Vert[[2, 3, 0], :] - Vert[[1, 2, 3], :], 2, 1)
    elif a.size == 2:
        EdgeLength = np.linalg.norm(Vert[[2, 3], :] - Vert[[1, 2], :], 2, 1)
    elif a.size == 1:
        EdgeLength = np.linalg.norm(Vert[0, :] - Vert[1, :])
        
    return EdgeLength - d


def CellVertCoor(a, Vert, F, Ang=0.0):
    """
    Calculate unknown vertices in a cell given known vertices and design vars a

    Parameters:
        a (ndarray): Optimization variables (1, 2, or 4).
        Vert (ndarray): (4, 3) array of partial vertex coordinates.
        F (callable): Surface interpolator function.
        Ang (float): Drape angle in degrees, used when a.size == 1.

    Returns:
        ndarray: Updated vertex array with all coordinates populated.
    """
    if a.size == 1:  # STEP 1
        Vert[1, 0:2] = Vert[0, 0:2] + a * CosSin(Ang + 90)
        Vert[1, 2] = F(Vert[1, 0], Vert[1, 1])
    if a.size >= 2:  # STEP 2 and 3
        Vert[2, 0:2] = Vert[1, 0:2] + a[0:2]
        Vert[2, 2] = F(Vert[1, 0] + a[0], Vert[1, 1] + a[1])
    if a.size == 4:  # STEP 2
        Vert[3, 0:2] = Vert[0, 0:2] + a[2:4]
        Vert[3, 2] = F(Vert[0, 0] + a[2], Vert[0, 1] + a[3])
    
    assert ~np.any(np.isnan(Vert)), 'Interpolant F is evaluated outside domain'
    return Vert

def CosSin(Alpha):
    """
    Return a unit vector in direction of Alpha.

    Parameters:
        Alpha (float): Angle in degrees.

    Returns:
        ndarray: 2-element unit vector.
    """
    return np.array([np.cos(Alpha*np.pi/180), np.sin(Alpha*np.pi/180)])

def ObjFun(a, Vert, F, d):
    """
    Objective function for shear minimization (Step 2 - standard solver). 
    Computes the sum of shear angles

    Parameters:
        a (ndarray): Design variables for vertex positioning.
        Vert (ndarray): Current vertex array.
        F (callable): Surface interpolator.
        d (float): Cell spacing.

    Returns:
        float: Sum of shear angles in degrees (deviation from 90°).
    """
    Vert = CellVertCoor(a, Vert, F)
    return np.sum(compute_cell_shear(Vert, 'standard'))  

def MoldCircIntersec(Vert, F, d, Ang, UnknownVertIdx):
    """
    Finds the location of a missing vertex on the mold surface by solving
    the intersection between a circle and the surface.
    
    Used for the NR solver.
    
    C, R: center and radius. Vec1, Vec2: perpend. unit vec. spanning circle
    
    Parameters:
        Vert (ndarray): (4, 3) array of partially defined cell vertices.
        F (callable): Surface interpolator.
        d (float): Cell edge length.
        Ang (float or []): Draping angle if UnknownVertIdx in (1, 3).
        UnknownVertIdx (int): Index (1–3) of vertex to compute.

    Returns:
        ndarray: Updated Vert with UnknownVertIdx assigned.
    """
    if UnknownVertIdx in (1, 3): # Step 1 (2nd node) and Step 2 iterative (Vert #4)
        # Circle is constructed along angle Ang, centered in Vert #1
        C = Vert[0,:]
        R = d
        Vec1 = np.array([-np.cos(Ang*np.pi/180), -np.sin(Ang*np.pi/180), 0])
        Vec2 = np.array([0, 0, 1])
    else: # Step 3 and Step 2 iterative (Vert # 3)
        # Circle is intersection of two spheres, centered in Vert 2 and Vert 4
        C = (Vert[1,:] + Vert[3,:])/2
        R = np.sqrt(d**2 - np.linalg.norm(C-Vert[1,:])**2)
        Vec1 = (Vert[0,:]-C)/np.linalg.norm(Vert[0,:]-C)
        CircAxis = (C-Vert[1,:])/np.linalg.norm(C-Vert[1,:])
        Vec2 = np.cross(Vec1,CircAxis)/np.linalg.norm(np.cross(Vec1,CircAxis))
    
    # Find the intersection between the circle and the surface using bisection 
    IntersecPt = np.array([np.nan, np.nan, np.nan])
    Theta = np.array([60*np.pi/180, (360-60)*np.pi/180])
    for i in range(100): # Max iter
        # Compute middle point
        Theta_mid = np.sum(Theta)/2
        # Circle pt. in 3D based on center, radius, 2 perp. vectors and angle
        CircCoor = C + R*np.cos(Theta_mid)*Vec1 + R*np.sin(Theta_mid)*Vec2
        FunVal_mid = CircCoor[2] - F(CircCoor[0],CircCoor[1])
        # Stop or adjust interval based on function value at midpoint
        if np.abs(FunVal_mid) < 5e-6: # Stopping tolerance
            IntersecPt = np.hstack((CircCoor[0:2].T,F(CircCoor[0],CircCoor[1])))
            break
        elif FunVal_mid > 0:
            Theta[0] = Theta_mid
        else:
            Theta[1] = Theta_mid
    # Store the IntersecPt in Vert at the row specified by UnknownVertIdx        
    Vert[UnknownVertIdx,:] = IntersecPt
    return Vert



def SphereCylinderIntersec(Vert, F, d, UnknownVertIdx, qd, R=0):
    """
    Sphere-cylinder intersection
    """
    
    # Step 3 and Step 2 iterative (Vert # 3)
    # Curve is the intersection of a sphere in vert 4/2 and a cylinder with axis in Vert 1 and Vert 2/4
    if qd == 0 or qd == 2:
        P1 = Vert[0,:]
        P2 = Vert[1,:]
        P4 = Vert[3,:]
    else:
        P1 = Vert[0,:]
        P2 = Vert[3,:]
        P4 = Vert[1,:]
    
    ## Step 1 - establish orthonormal basis
    # compute a hat
    cylinder_axis = P2 - P1
    L_axis = np.linalg.norm(cylinder_axis)
    if L_axis < 1e-12:
        raise ValueError("Cylinder axis is degenerate: P1 and P2 coincide.")
    a_hat = cylinder_axis / L_axis
    
    # compute u hat
    t0 = np.dot(P4 - P1, a_hat)
    P0 = P1 + t0 * a_hat
    u_hat = (P4 - P0)/ np.linalg.norm(P4 - P0)
    
    # compute v hat
    v_hat = np.cross(a_hat,u_hat)  
    
    ## Step 2 - variables
    a = np.linalg.norm(P0 - P4) # Sphere centre offset from cylinder (should be d)
    
    if R>0: 
        b = (R**2 - d**2 - a**2) / (2 * a)
    else:
        b = - a**2 / (2 * a) # Special case due to radius of sphere and cylinder is identical
    
    ## Step 3 - bisection
    # Find the intersection between the curve and the surface using bisection 
    IntersecPt = np.array([np.nan, np.nan, np.nan])

    phi = np.array([-60*np.pi/180, +20*np.pi/180])
    
    x_local = d * np.cos(phi[0]+0.01)
    y_local = d * np.sin(phi[0]+0.01)
    if 2*a * (b + d*np.cos(phi[0]+0.01)) > 0:
        z_local = np.sqrt(2*a * (b + d*np.cos(phi[0]+0.01)))
    else:
        z_local = 0
    CurveCoor = P0 + x_local*u_hat + y_local*v_hat + z_local*a_hat
    FunVal_beginning = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
    
    x_local = d * np.cos(phi[1]-0.01)
    y_local = d * np.sin(phi[1]-0.01)
    if 2*a * (b + d*np.cos(phi[1]-0.01)) > 0:
        z_local = np.sqrt(2*a * (b + d*np.cos(phi[1]-0.01)))
    else:
        z_local = 0
    CurveCoor = P0 + x_local*u_hat + y_local*v_hat + z_local*a_hat
    FunVal_ending = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
    
    increasing = FunVal_ending >  FunVal_beginning
    
    for i in range(100): # Max iter
        # Compute middle point
        phi_mid = np.sum(phi)/2
        # Curve pt. in 3D based on center, radius, 2 perp. vectors and angle
        x_local = d * np.cos(phi_mid)
        y_local = d * np.sin(phi_mid)
        if 2*a * (b + d*np.cos(phi_mid)) > 0:
            z_local = np.sqrt(2*a * (b + d*np.cos(phi_mid)))
        else:
            z_local = 0
        CurveCoor = P0 + x_local*u_hat + y_local*v_hat + z_local*a_hat
        FunVal_mid = CurveCoor[2] - F(CurveCoor[0],CurveCoor[1])
        # Stop or adjust interval based on function value at midpoint
        if np.abs(FunVal_mid) < 5e-6: # Stopping tolerance
            IntersecPt = np.hstack((CurveCoor[0:2].T,F(CurveCoor[0],
                                                       CurveCoor[1])))
            break
        elif np.isnan(FunVal_mid): 
            phi[0] = phi_mid
        elif FunVal_mid < 0 and increasing: # Changed sign on this one
            phi[0] = phi_mid
        elif FunVal_mid > 0 and not increasing: # Changed sign on this one
            phi[0] = phi_mid
        else:
            phi[1] = phi_mid
            
        if i == 99: # Last iteration
            IntersecPt = np.hstack((CurveCoor[0:2].T,F(CurveCoor[0],
                                                       CurveCoor[1])))
    # Store the IntersecPt in Vert at the row specified by UnknownVertIdx        
    Vert[UnknownVertIdx,:] = IntersecPt
    return Vert


def NRSol(Vert, CellAng, F, d, PreShear, i):
    """
    Newton-Raphson solver to find CellAng that min. shear in cell (Step 2).

    Parameters:
        Vert (ndarray): Partial cell with known vertices.
        CellAng (float): Initial cell angle.
        F (callable): Surface function.
        d (float): Cell spacing.
        PreShear (float): Prescribed average shear target.
        i (int): Quadrant index.

    Returns:
        tuple: (Vert, CellAng) updated vertex array and converged angle.
    """
    for j in range(100): # Max iter
        # Calculate the current objective and the vertex coord. Check converg.
        Obj_curr, Vert = Step2Obj(CellAng,Vert,F,d,i,PreShear)
        if np.abs(Obj_curr) < 1e-3 or j == 100:
            break
        # Calculate a perturbed objective and a forward finite diff. gradient
        Obj_pert = Step2Obj(CellAng+1e-8,Vert,F,d,i,PreShear)[0] 
        Grad = (Obj_pert - Obj_curr)/1e-8
        # Update the design variable using a scaled step
        CellAng = CellAng - 0.5*(Obj_curr / Grad)
    return Vert, CellAng

def Step2Obj(CellAng, Vert, F, d, i, PreShear):
    """
    Compute all cell vertices given the angle CellAng of cell edge 1-4.
    Place: node #4 based on CellAng and d, and node #3 based on d (kinematics)

    Parameters:
        CellAng (float): Cell angle guess.
        Vert (ndarray): Current vertex array.
        F (callable): Surface function.
        d (float): Cell spacing.
        i (int): Quadrant index.
        PreShear (float): Target average shear.

    Returns:
        tuple: (Objective value, updated Vert)
    """
    Vert = MoldCircIntersec(Vert,F,d,CellAng,3)
    Vert = MoldCircIntersec(Vert,F,d,[],2)
    
    # Evaluate shear and calculate the objective
    Shear = compute_cell_shear(Vert,'NR',i)
    Obj = np.sum(Shear - PreShear)/4
    return Obj, Vert   