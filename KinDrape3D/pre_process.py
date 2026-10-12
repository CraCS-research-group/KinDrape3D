"""
pre_process.py

Module for pre-processing tool geometries and setting up surfaces
used in kinematic draping simulations.

Includes:
- Parametric surface generation (ramp, hemisphere)
- Point cloud interpolation


Authors: Peter Hede Broberg, Christian Krogh, Bastian Schäfer, Jan Paul Wank, 
         Luise Kärger, Brian Lau Verndal Bak, Esben Lindgaard
"""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator as CT2DInt
from scipy.interpolate import griddata
import warnings
import meshio

# -------------------------------------------------------------------------
# Functions for fabric and draping initializations
# -------------------------------------------------------------------------

def setup_grid_and_initials(mesh, initial_conditions):
    """
    Prepare grid, discretisation, start data, and (optional) trimming masks.

    Parameters
    ----------
    mesh : dict
        Mesh definition.
    initial_conditions : dict
        Initial condition definition.

    Returns
    -------
    setup : dict
        Dictionary with stable keys:
        - "Grid" : array-like
        - "d" : float
        - "OrgNode" : array-like
        - "Org" : array-like
        - "Ang" : float
        - "masks" : dict with keys
            - "node_outside_mask" : array-like or None
            - "cell_inside_mask_1d" : array-like or None
    """
    # --- Define variables ---------------------------------------------------
    L = mesh["L"]
    W = mesh["W"]
    d_in = mesh["d"]
    orientation = mesh["orientation"]

    start_xy = initial_conditions["drape_start_mould_xy"]
    start_us = initial_conditions["drape_start_fabric_us"]
    Ang = initial_conditions["ang"]

    # --- Orientation-aware setup ------------------------------------------
    if orientation != 0: # If the fabric is oriented
        Grid, d, OrgNode, Org, Ang = preprocess_superset(
            L, W, d_in, orientation,
            start_xy,   # (x_m, y_m)
            start_us,   # (U, V) from fabric lower-left
            Ang,        # intended lay direction on mould
            margin_cells=0,
        )

        node_outside_mask, cell_inside_mask_1d = (
            compute_superset_masks(Grid, d, orientation, L, W, OrgNode, Org,
                require_all_nodes=True, )
        )

    else:  # If the fabric is not oriented other than 0
        Grid, d = fabric_size_to_grid(
            L, W, d_in, dimensions_driving=False
        )
        OrgNode, Org = startpoints_to_initials(
            start_xy, start_us, d, Grid, Ang
        )
        node_outside_mask = None
        cell_inside_mask_1d = None

    return {
        "Grid": Grid,
        "d": d,
        "OrgNode": OrgNode,
        "Org": Org,      # mould XY coordinates
        "Ang": Ang,
        "masks": {
            "node_outside_mask": node_outside_mask,
            "cell_inside_mask_1d": cell_inside_mask_1d,
        },
    }

def startpoints_to_initials(drape_start_mould_xy, drape_start_fabric_us, d, 
                            grid, Ang):
    """
    Compute solver 'org_node' and 'org' from user start points.

    Parameters
    ----------
    drape_start_mould_xy : array-like of length 2
        2D start location on the mould plane used by the solver (x, y).
    drape_start_fabric_us : array-like of length 2
        2D start location on the fabric measured from the LOWER-LEFT corner 
        (U to right, V up)
    d : float
        Discretization distance (cell size).
    grid : list/tuple of two ints
        [rows, cols] = number of grid nodes in U (length) and V (width).

    Returns
    -------
    org_node : list [row, col]
        Lower-left node of the cell that contains the drape_start_fabric point.
        Uses bottom-left origin, i.e., row increases to the right, col upwards.
    org : list [x, y]
        Origin on the mould plane such that the lower-left node of that cell
        is positioned consistently with the fabric offset. Specifically, the
        continuous fabric point (U, V) maps to drape_start_mould_xy.

    Notes
    -----
    - Node indexing:
        cell lower-left node indices are (row=i0, col=j0) with
            i0 = floor(U/d), j0 = floor(V/d).
      We clamp i0 ∈ [0, cols-2], j0 ∈ [0, rows-2] so the cell exists.
    - Positioning convention:
        If node (i, j) maps to the 2D mould plane coordinate:
            X(i,j) = org_x + i*d
            Y(i,j) = org_y + j*d
        then the continuous fabric point (U, V) maps to:
            (org_x + U, org_y + V).
        To place the start point exactly at drape_start_mould_xy, we set:
            org = drape_start_mould_xy - (U, V).
    """
    rows, cols = int(grid[0]), int(grid[1])

    U, V = float(drape_start_fabric_us[0]), float(drape_start_fabric_us[1])
    x_m, y_m = float(drape_start_mould_xy[0]), float(drape_start_mould_xy[1])

    if d <= 0:
        raise ValueError("Discretization distance d must be positive.")
    if rows < 2 or cols < 2:
        raise ValueError("Grid must have at least 2 nodes in each direction.")
    if U < 0 or V < 0:
        raise ValueError(
            "drape_start_fabric_us must be measured from the lower-left "
            "corner (non-negative)."
            )

    # Indices of the lower-left node of the containing cell
    i0 = int(np.floor(U / d))
    j0 = int(np.floor(V / d))

    # Clamp to ensure the cell exists within the grid
    i0 = max(0, min(i0, rows - 2))
    j0 = max(0, min(j0, cols - 2))

    org_node = [i0, j0]

    # Fabric coordinates of lower-left node of the containing cell
    u_node = i0 * d
    v_node = j0 * d

    # Origin on mould so that the (U,V) fabric point maps to the start point. 
    translation_x = (U - u_node) 
    translation_y = (V - v_node) 
    org_x = x_m - translation_x 
    org_y = y_m - translation_y 
    
    if ([abs(translation_x/d), abs(translation_y/d)] > [0.75] 
            or [abs(translation_x/d), abs(translation_y/d)] < [0.25]):
        warnings.warn(
            "The start point is close to the cell edges. The geodesic lines"
            " should be compute in cell centre for best performance. To "
            "alleviate this, change the discretization d."
            )

    org = [org_x, org_y]
    
    c, s = np.cos(np.radians(Ang)), np.sin(np.radians(Ang))
    R = np.array([[ c, -s],
                  [ s,  c]])
    
    org = R @ org

    return org_node, org

def fabric_size_to_grid(L, W, d, dimensions_driving=False):
    """
    Convert fabric dimensions and discretization spacing to solver grid size.

    Parameters
    ----------
    L : float
        Fabric length along the warp direction.
    W : float
        Fabric width along the weft direction.
    d : float
        Desired cell spacing.
    dimensions_driving : bool, optional
        If False (default): 
            d is fixed, grid size is rounded to nearest whole number of cells.
        If True: 
            The fabric dimensions L and W are fixed, d may be adjusted slightly
            so the fabric length (and approximately width) is exactly satisfie

    Returns
    -------
    grid : list of int
        [rows, cols] = number of grid nodes (warp, weft).
    d_eff : float
        Effective discretization distance actually used.
    """

    # --- Basic checks
    if d <= 0:
        raise ValueError("Discretization distance d must be positive.")
    if L <= 0 or W <= 0:
        raise ValueError("Fabric dimensions L and W must be positive.")

    if not dimensions_driving:
        # d is fixed; adjust grid to nearest integer
        rows = int(round(L / d)) + 1
        cols = int(round(W / d)) + 1
        d_eff = d

    else:
        # L and W are fixed; adjust d slightly to make them fit exactly
        rows = int(round(L / d)) + 1
        cols = int(round(W / d)) + 1
        d_eff_L = L / (rows - 1)
        d_eff_W = W / (cols - 1)
        # Pick the smaller to ensure the entire fabric fits or use mean
        d_eff = min(d_eff_L, d_eff_W)

    return [rows, cols], d_eff

def compute_superset_masks(grid_sup, d_eff, alpha_deg, L, W, org_node, org,
                           require_all_nodes=True):
    """
    Build masks in the UNDEFORMED superset plane.

    Inputs
    ------
    grid_sup : [rows_sup, cols_sup]
    d_eff : float
    alpha_deg : float
    L, W : float
    org_node : [row, col]  (lower-left node of start cell, superset indices)
    org : [x0, y0]         (solver origin)
    require_all_nodes : bool
        If True: a cell is 'inside' only if all 4 corner nodes are inside.
        If False: a cell is 'inside' if any of its 4 corner nodes are inside.

    Returns
    -------
    node_outside_mask : (rows_sup, cols_sup) bool
        True where node is OUTSIDE polygon; False where inside/on-edge.
    cell_inside_mask_1d : ( (rows_sup-1)*(cols_sup-1), ) bool
        Row-major flattening with cell 0 at bottom-left, then rightwards,
        then next row upwards.
    corners_rot : (5,2) array
        Closed polygon used for the test (rotated rectangle about center).
    """

    rows_sup, cols_sup = int(grid_sup[0]), int(grid_sup[1])

    # translation vector t that places the grid in the mould XY frame
    b_vec = np.array([org_node[0] * d_eff, org_node[1] * d_eff])
    c_vec = np.array(org, dtype=float)
    t_vec = c_vec - b_vec  # so node(i,j) sits at (t_x + i*d, t_y + j*d)

    # node coordinate arrays in undeformed superset plane
    X_sup = np.arange(cols_sup) * d_eff + t_vec[0]  # x uses column index
    Y_sup = np.arange(rows_sup) * d_eff + t_vec[1]  # y uses row index

    # rotated fabric polygon (about fabric center) placed at superset center
    rad = np.deg2rad(alpha_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s,  c]])
    corners = np.array([
        [-L/2, -W/2],
        [-L/2,  W/2],
        [ L/2,  W/2],
        [ L/2, -W/2],
        [-L/2, -W/2]
    ], dtype=float)
    T = np.array([t_vec[0] + (rows_sup - 1) * d_eff / 2.0,
                  t_vec[1] + (cols_sup - 1) * d_eff / 2.0], dtype=float)
    corners_rot = T + corners @ R.T  # (5,2), closed

    # --- node mask: outside = True ---
    node_outside_mask = np.zeros((rows_sup, cols_sup), dtype=bool)
    for j in range(rows_sup):         # row index -> y
        y = Y_sup[j]
        for i in range(cols_sup):     # col index -> x
            x = X_sup[i]
            inside = _point_in_poly(x, y, corners_rot)
            node_outside_mask[j, i] = not inside

    # --- cell mask (2D), then flatten to 1D in requested order ---
    cell_inside_mask_2d = np.zeros((rows_sup - 1, cols_sup - 1), dtype=bool)
    for j in range(rows_sup - 1):     # cell row (bottom to top)
        for i in range(cols_sup - 1): # cell col (left to right)
            # corner nodes: (j,i), (j,i+1), (j+1,i+1), (j+1,i)
            n00 = not node_outside_mask[j,   i  ]
            n10 = not node_outside_mask[j,   i+1]
            n11 = not node_outside_mask[j+1, i+1]
            n01 = not node_outside_mask[j+1, i  ]
            if require_all_nodes:
                cell_inside = n00 and n10 and n11 and n01
            else:
                cell_inside = n00 or n10 or n11 or n01
            cell_inside_mask_2d[j, i] = cell_inside

    # Flatten with cell 0 at bottom-left, then rightwards, then upwards:
    # flat_idx = j*(cols_sup-1) + i
    cell_inside_mask_1d = cell_inside_mask_2d.reshape(-1, order='C')

    return node_outside_mask, cell_inside_mask_1d

def preprocess_superset(
    L, W, d, alpha_deg,
    drape_start_mould_xy,      # (x_m, y_m)
    drape_start_fabric_us,     # (u, s) from fabric lower-left
    theta_deg,                 # intended lay direction on mould
    margin_cells=2,
    dimensions_driving=False,
    theta_is_fabric_u=True
):
    """
    Build a superset grid for a fabric rotated by alpha_deg (rotation about 
    fabric CENTER).
    Returns solver-ready: grid_sup, d_eff, org_node, org, ang_sup.
    Conventions: rows↔Length(X,U), cols↔Width(Y,V), bottom-left origin.
    """

    if d <= 0: raise ValueError("d must be positive.")
    if L <= 0 or W <= 0: raise ValueError("L and W must be positive.")
    U, S = map(float, drape_start_fabric_us)
    if U < 0 or S < 0: raise ValueError("drape_start_fabric_us must be ≥ 0.")
    x_m, y_m = map(float, drape_start_mould_xy)

    # Node counts for the unrotated fabric (used for d_eff if dimensions drive)
    rows0 = int(round(L / d)) + 1
    cols0 = int(round(W / d)) + 1
    if rows0 < 2 or cols0 < 2:
        raise ValueError("Grid must have at least 2 nodes per direction.")
    
    min_spacing = min(L / (rows0 - 1), W / (cols0 - 1))
    d_eff = min_spacing if dimensions_driving else float(d)

    # Axis-aligned bbox of the rotated rectangle (rotation about center)
    a = np.deg2rad(alpha_deg)
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(a), np.sin(a)
    Lsup = abs(L * c) + abs(W * s)   # extent along X/rows
    Wsup = abs(L * s) + abs(W * c)   # extent along Y/cols

    rows_sup = int(np.ceil(Lsup / d_eff)) + 1 + 2 * margin_cells
    cols_sup = int(np.ceil(Wsup / d_eff)) + 1 + 2 * margin_cells
    rows_sup = max(rows_sup, 2)
    cols_sup = max(cols_sup, 2)
    grid_sup = [rows_sup, cols_sup]

    # --- Fabric->superset mapping (rotation about center) ---
    # Center of the superset "usable area" after margin
    sup_L = (rows_sup - 1) * d_eff   # total length span in superset
    sup_W = (cols_sup - 1) * d_eff
    C = np.array([0.5*sup_L, 0.5*sup_W]) # Center of the set

    # Rotation matrix
    R = np.array([[ c, -s],
                  [ s,  c]])

    # Map a fabric point (U,V) (measured from lower-left) to superset (û, v̂):
    # 1) shift to fabric center, 2) rotate, 3) shift to superset center C
    def us_to_hat(US):
        us_centered = np.array(US) - np.array([L/2, W/2])
        return C + R @ us_centered

    # Superset coordinates of the *continuous* fabric start point
    uhat, shat = us_to_hat([U, S])

    # Lower-left node (cell origin) that contains (û, v̂)
    i0 = int(np.floor(uhat / d_eff))      # along Length (rows)
    j0 = int(np.floor(shat / d_eff))      # along Width  (cols)
    i0 = max(0, min(i0, rows_sup - 2))
    j0 = max(0, min(j0, cols_sup - 2))
    org_node = [i0, j0]

    # Solver origin so that org + (û, v̂) == drape_start_mould_xy
    uhat_node = i0 * d_eff
    shat_node = j0 * d_eff
    
    translation_x = (uhat - uhat_node)
    translation_y = (shat - shat_node) 
    
    org_x = x_m - translation_x
    org_y = y_m - translation_y
    org = [org_x, org_y]
    
    if ([abs(translation_x/d), abs(translation_y/d)] > [0.75] 
            or [abs(translation_x/d), abs(translation_y/d)] < [0.25]):
        warnings.warn(
            "The start point is close to the cell edges. The geodesic lines"
            " should be compute in cell centre for best performance. To "
            "alleviate this, change the discretization d."
            )
    
    if theta_is_fabric_u:
        c, s = np.cos(theta_rad-a), np.sin(theta_rad-a)
    else: 
        c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[ c, -s],
                  [ s,  c]])
    
    org = R @ org

    # Initial drape direction
    drape_dir = float(theta_deg) - float(alpha_deg)
    ang_sup = drape_dir if theta_is_fabric_u else float(theta_deg)

    return grid_sup, d_eff, org_node, org, ang_sup


def _point_in_poly(x, y, poly_xy): # May be vectorized for performance
    """Point-in-polygon test. poly_xy must be closed or will be closed here."""
    poly = np.asarray(poly_xy, dtype=float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    inside = False
    for k in range(len(poly) - 1):
        x1, y1 = poly[k]
        x2, y2 = poly[k + 1]
        # Check edge intersection with horizontal ray
        if ((y1 <= y < y2) or (y2 <= y < y1)):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-16)
            if x <= x_int + 1e-16:
                inside = not inside
        # On-edge check (tolerance)
        denom = (x2 - x1)**2 + (y2 - y1)**2
        if denom > 0:
            t = ((x - x1)*(x2 - x1) + (y - y1)*(y2 - y1)) / denom
            if 0.0 <= t <= 1.0:
                xp = x1 + t*(x2 - x1); yp = y1 + t*(y2 - y1)
                if abs(x - xp) <= 1e-12 and abs(y - yp) <= 1e-12:
                    return True
    return inside


# -------------------------------------------------------------------------
# Functions for tool initializations
# -------------------------------------------------------------------------

def tool_definition(tool):
    """
    Constructs a surface interpolator F(x, y) and corresponding mesh grid 
    (X, Y, Z) from a tool definition dictionary.

    Parameters
    ----------
    tool: dict
        Tool definition

    Returns
    -------
    (F, X, Y, Z) : tuple
        F is an interpolator and X, Y, Z are mesh grids.
    """
    if tool['tool_type'] == 'ramp':
        X = tool['x_tool']
        Y = tool['y_tool']
        Z = ramp_geometry(X, tool['tool_parameters'])
        F = CT2DInt((X.ravel(), Y.ravel()), Z.ravel(), fill_value=np.min(Z))

    elif tool['tool_type'] == 'hemisphere':
        tool_parameters = tool['tool_parameters']
        The = tool['x_tool']
        Phi = tool['y_tool']
        # Spherical to Cartesian conversion
        X = tool_parameters['radius'] * np.cos(The) * np.sin(Phi)
        Y = tool_parameters['radius'] * np.sin(The) * np.sin(Phi)
        Z = tool_parameters['radius'] * np.cos(Phi)
        F = CT2DInt((X.ravel(), Y.ravel()), Z.ravel(), fill_value=np.min(Z))
        
    elif tool['tool_type'] == 'hemisphere_fillet':
        X = tool['x_tool']
        Y = tool['y_tool']
        Z = z_mould_hemisphere_fillet(X, Y, tool['tool_parameters']['R'], 
                                      tool['tool_parameters']['r_f'])
        F = make_mould_F(R= tool['tool_parameters']['R'], 
                         rf= tool['tool_parameters']['r_f'])

    elif tool['tool_type'] == 'point_cloud':
        if tool['file_name'].endswith(".msh"):
            mesh = meshio.read(tool['file_name'])
            data = mesh.points 
        elif tool['file_name'].endswith(".XYZ"):
            data = np.loadtxt(tool['file_name']) * 1e-3
        else:
            raise ValueError(
                "Only .msh and .XYZ files supported for point cloud."
                )

        if data.shape[1] != 3:
            raise ValueError("Each line must contain exactly three values:",
                             "X Y Z")

        # Assign to X, Y, and Z
        X, Y, Z = data[:, 0], data[:, 1], data[:, 2] 

        if tool['centre']:
            X -= (np.max(X) + np.min(X)) / 2
            Y -= (np.max(Y) + np.min(Y)) / 2

        X, Y, Z = make_meshgrid_from_unordered(X, Y, Z)
        F = CT2DInt((X.ravel(), Y.ravel()), Z.ravel(), fill_value=np.min(Z))

    elif tool['tool_type'] == 'double_curved':
        X = tool['x_tool']
        Y = tool['y_tool']
        F = lambda x,y: 1.004*x + 1.089*y - 3.667*x**2 -4.4*x*y - 3.75*y**2 + \
            3.086*x**3 + 8.889*x**2*y + 4.321*y**3; 
        Z = F(X,Y)


    return F, X, Y, Z

def make_mould_F(R, rf):
    def F(x, y):
        return z_mould_hemisphere_fillet(x, y, R, rf)
    return F

def z_mould_hemisphere_fillet(x, y, R, rf):
    """
    Convex hemisphere with a circular fillet down to a flat flange (z=0).

    Parameters
    ----------
    x, y : float or ndarray
        Cartesian coordinates.
    R : float
        Hemisphere radius (R > 0).
    rf : float
        Fillet radius (0 < rf < R is a sensible range).

    Returns
    -------
    z : float or ndarray
        Height of the mould surface at (x, y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt(x * x + y * y)

    # Join height from external tangency of the two circles in (r,z)
    z_join = (R * rf) / (R + rf)
    r_join = np.sqrt(R * R - z_join * z_join)

    # Fillet circle centre (r_c, z_c) = (r_c, rf)
    r_c = np.sqrt((R + rf) * (R + rf) - rf * rf)

    z = np.zeros_like(r)

    # Sphere region
    m_sphere = r <= r_join
    z[m_sphere] = np.sqrt(R * R - r[m_sphere] * r[m_sphere])

    # Fillet region (upper half of the fillet circle, which hit z=0 at r = r_c)
    m_fillet = (r > r_join) & (r <= r_c)
    dr = r[m_fillet] - r_c
    z[m_fillet] = rf - np.sqrt(np.maximum(0.0, rf * rf - dr * dr))

    # Flange region stays at z = 0
    return float(z) if z.shape == () else z

def ramp_geometry(x, tool_parameters):
    """
    Computes ramp geometry surface height Z based on x-coordinate and geometry 
    parameters.

    Args:
        x (ndarray): x-coordinates of the mesh grid.
        tool_parameters (dict): Dictionary with keys 'l_flanges', 'h_ramp',
                                'l_ramp', 'r_transition'.

    Returns:
        ndarray: z-values corresponding to the ramp surface.
    """
    
    # Unpack parameters
    l_flanges = tool_parameters['l_flanges']
    h_ramp = tool_parameters['h_ramp']
    l_ramp = tool_parameters['l_ramp']
    r_transition = tool_parameters['r_transition']

    x = np.asarray(x)
    original_shape = x.shape
    x_flat = x.ravel()

    theta_ramp = np.arctan(h_ramp / l_ramp)
    z0 = 0  # Initial flat section height

    # Define transition segment boundaries
    x1 = l_flanges - np.tan(theta_ramp / 2) * r_transition
    x2 = x1 + np.sin(theta_ramp) * r_transition
    x3 = l_flanges + l_ramp - ((np.sin(theta_ramp) - np.tan(theta_ramp / 2)) 
                               * r_transition)
    x4 = x3 + np.sin(theta_ramp) * r_transition

    a = -h_ramp / l_ramp  # Linear ramp slope
    b = z0 - a * l_flanges
    z3 = a * x3 + b

    # Allocate z-values and fill per region
    z_flat = np.full_like(x_flat, np.nan, dtype=np.float64)

    mask1 = x_flat < x1
    z_flat[mask1] = z0

    mask2 = (x_flat >= x1) & (x_flat < x2)
    theta = np.arcsin((x_flat[mask2] - x1) / r_transition)
    z_flat[mask2] = z0 - (1 - np.cos(theta)) * r_transition

    mask3 = (x_flat >= x2) & (x_flat < x3)
    z_flat[mask3] = a * x_flat[mask3] + b

    mask4 = (x_flat >= x3) & (x_flat < x4)
    theta = np.arcsin((x4 - x_flat[mask4]) / r_transition)
    z_flat[mask4] = z3 - ((1 - np.cos(theta_ramp)) - (1 - np.cos(theta))
                          ) * r_transition

    mask5 = x_flat >= x4
    z_flat[mask5] = z0 - h_ramp

    return z_flat.reshape(original_shape) 

def make_meshgrid_from_unordered(X,Y,Z):
    """
    Converts unordered 1D point cloud arrays (X, Y, Z) into regular mesh grids.

    Args:
        X, Y, Z (ndarray): Flattened point cloud data.

    Returns:
        tuple: (grid_x, grid_y, grid_z) regular grid for surface interpolation.
    """
    # Define grid resolution
    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(X), np.max(X), 100),
        np.linspace(np.min(Y), np.max(Y), 100)
    )
    
    # Interpolate Z values onto the grid
    grid_z = griddata(
        points=(X, Y),
        values=Z,
        xi=(grid_x, grid_y),
        method='linear', # 'cubic' or 'nearest' also available
        fill_value=np.min(Z)
    )

    return grid_x, grid_y, grid_z