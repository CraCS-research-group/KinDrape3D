# -*- coding: utf-8 -*-
"""
utility.py

Utility functions for visualization and geometry generation in the KinDrape3D 
package.

This module provides tools to:
- Plot kinematic draping results, including surface and cell-wise shear maps.

Functions:
- plot_drape_results: Visualizes shear fields and 3D fiber placement.

Authors: Peter Hede Broberg, Christian Krogh, Bastian Schäfer, Jan Paul Wank, 
         Luise Kärger, Brian Lau Verndal Bak, Esben Lindgaard
"""

import numpy as np
from mpl_toolkits.mplot3d import (Axes3D, art3d)
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator as CT2DInt
import json
from pathlib import Path

# -------------------------------------------------------------------------
# Functions for saving and loading results
# -------------------------------------------------------------------------
def _to_jsonable(x):
    """
    Convert numpy objects (arrays + scalars) into JSON-friendly Python types.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def load_drape_results(study_name, run_id, base_dir="results"):
    run_dir = Path(base_dir) / study_name / run_id
    fields = dict(np.load(run_dir / "fields.npz", allow_pickle=True))
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return run_dir, fields, meta

def save_drape_results(fields, settings, timing_seconds, fig, ax, study_name, 
                       run_id, save_figures=True,):
    out_dir = Path("results") / study_name / run_id
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    arrays = {k: np.asarray(v) for k, v in fields.items()}
    np.savez_compressed(out_dir / "fields.npz", **arrays)

    meta = {
        "study_name": study_name,
        "run_id": run_id,
        "timing_process_time_s": float(timing_seconds),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "settings": _to_jsonable(settings),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), 
                                       encoding="utf-8")

    lines = [
        f"study_name: {study_name}",
        f"run_id: {run_id}",
        f"process_time_s: {timing_seconds:.6f}",
        "",
        "saved_arrays:",
        *[f"  - {k}: {tuple(v.shape)}" for k, v in arrays.items()],
        "",
        "settings_keys:",
        *[f"  - {k}" for k in sorted(settings.keys())],
    ]
    (out_dir / "overview.txt").write_text("\n".join(lines) + "\n", 
                                          encoding="utf-8")
   
    def _save(stem):
        fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    
    if not save_figures:
        return
    
    _save("figure_default")

    if hasattr(ax, "view_init") and hasattr(ax, "elev") and hasattr(ax, 
                                                                    "azim"):
        elev0, azim0 = float(ax.elev), float(ax.azim)
        views = {"xy": (90, -90), "xz": (0, -90), "yz": (0, 0)}

        for name, (elev, azim) in views.items():
            ax.view_init(elev=elev, azim=azim)
            _save(f"figure_{name}")

        ax.view_init(elev=elev0, azim=azim0)

def save_results(array, name):
    if array.ndim < 3:
        np.savetxt(name+'.txt', array)
    else:
        # Save the shape so we can reconstruct later
        original_shape = array.shape
        
        # Flatten and save to text
        np.savetxt(name+'_flat.txt', array.reshape(-1))
        
        # Also save the shape to a separate file (optional but helpful)
        with open(name+'_shape.txt', 'w') as f:
            f.write(' '.join(map(str, original_shape)))
            
def load_results(name, less_than_3D):
    if less_than_3D:
        array = np.loadtxt(name+'.txt')
    else:
        with open(name+'_shape.txt', 'r') as f:
            shape = tuple(map(int, f.read().split()))
        
        # Load the flattened array
        flat_array = np.loadtxt(name+'_flat.txt')
        
        # Reshape it back
        array = flat_array.reshape(shape)
    return array

# -------------------------------------------------------------------------
# Functions for plotting results
# -------------------------------------------------------------------------
def plot_fibres(fiber_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for fiber in fiber_data:
        fiber_array = np.array(fiber)
        # Extract x, y, and z coordinates
        x = fiber_array[:, 0]
        y = fiber_array[:, 1]
        z = fiber_array[:, 2]
        
        # Plot the fiber as a line
        ax.plot3D(x, y, z)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Plot of Fibers')

def plot_drape_results(field, P, X, Y, Z, *, label="Shear angle [deg]", 
                       preform=False, P_2=None, connect_bars=None,):
    """
    Plot a 3D kinematic draping result on a tool surface.

    Parameters
    ----------
    field : ndarray
        Cell-wise scalar field with shape (n_cells,) or (n_cells, 4).
        If (n_cells, 4), the function plots the mean value per cell.
    P : list
        List of 4-vertex polygons (each is a list/array of 3D points) for the
        bottom layer cells.
    X, Y, Z : ndarray
        Tool surface mesh grids. If X is not 2D, the function interpolates to
        a regular grid for surface plotting.
    label : str, optional
        Colourbar label. Default is "Shear angle [deg]".
    preform : bool, optional
        If True, also plots a second layer (P_2) and connecting bars.
    P_2 : list, optional
        Polygons for the top layer cells (same structure as P). Required if
        preform is True.
    connect_bars : list, optional
        List of 2-point segments connecting layers. Each item is (pt1, pt2),
        where pt1 and pt2 are 3D points. Required if preform is True.

    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D axis handle.
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    if X.ndim != 2:
        F = CT2DInt((X.ravel(), Y.ravel()), Z.ravel(), fill_value=np.min(Z))
        X_grid, Y_grid = np.meshgrid(
            np.linspace(np.min(X), np.max(X), 100),
            np.linspace(np.min(Y), np.max(Y), 100),
        )
        Z_grid = F(X_grid, Y_grid)
        X, Y, Z = X_grid, Y_grid, Z_grid

    # --- Reduce field to one value per cell ---------------------------------
    field = np.asarray(field, dtype=float)
    field_plot = (
        np.nanmean(field, axis=1) if field.ndim == 2 else field.reshape(-1)
    )

    valid = np.isfinite(field_plot)
    if np.any(valid):
        vmin = float(np.nanmin(field_plot[valid]))
        vmax = float(np.nanmax(field_plot[valid]))
        # Avoid zero-span colour scaling
        if (vmax - vmin) < 1e-12:
            mid = 0.5 * (vmin + vmax)
            pad = 5e-13
            vmin, vmax = mid - pad, mid + pad
    else:
        vmin, vmax = 0.0, 1.0

    cmap_name = "jet"
    cmap = mpl.colormaps.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    rgba = cmap(norm(field_plot))
    if np.any(~valid):
        rgba[~valid] = (0.7, 0.7, 0.7, 1.0)  # grey for NaNs

    # --- Plot surface + cell polygons ---------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=0)
    fig.add_axes(ax)

    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color=(0.6, 0.7, 0.8), 
                    alpha=0.4)

    pc = art3d.Poly3DCollection(P, cmap=cmap_name)
    pc.set_facecolor(rgba.tolist())
    pc.set_edgecolor("k")
    ax.add_collection3d(pc)

    # --- Optional preform overlay -------------------------------------------
    if preform:
        if P_2 is None:
            raise ValueError('preform=True requires "P_2".')
        if connect_bars is None:
            raise ValueError('preform=True requires "connect_bars".')

        pc_bars = art3d.Poly3DCollection(connect_bars)
        pc_bars.set_edgecolor("k")
        ax.add_collection3d(pc_bars)

        pc_2 = art3d.Poly3DCollection(P_2, cmap=cmap_name)
        pc_2.set_facecolor(rgba.tolist())
        pc_2.set_edgecolor("k")
        ax.add_collection3d(pc_2)


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))

    fig.colorbar(
        pc,
        shrink=0.75,
        boundaries=np.linspace(vmin, vmax, 50),
        values=np.linspace(0, 1, 49),
        label=label,
    )

    return ax, fig

def find_outer_contour(node):
    nr, nc, _ = node.shape
    outline = np.concatenate(
        [
            node[0, :, :],          # top edge (left → right)
            node[1:-1, -1, :],      # right edge (top → bottom, skip corners)
            node[-1, ::-1, :],      # bottom edge (right → left)
            node[-2:0:-1, 0, :],    # left edge (bottom → top, skip corners)
        ],
        axis=0,
    )
    outline = np.vstack([outline, outline[0]])          # close outline
    return outline[:, :2]                      


def find_masked_outer_contour(node, mask):
    """
    Find outline nodes for a masked fabric.

    Parameters
    ----------
    node : ndarray
        Array of node coordinates with shape (nx, ny, 3).
    mask : ndarray
        Boolean array with shape (nx, ny).
        False = inside fabric
        True  = outside fabric

    Returns
    -------
    outline_nodes : ndarray
        Array of outline node coordinates with shape (n_outline, 3).
    outline_idx : ndarray
        Array of outline indices with shape (n_outline, 2), giving [i, j].
    """
    nx, ny = mask.shape
    outline_idx = []

    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                continue

            # Boundary of array counts as edge
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                outline_idx.append([i, j])
                continue

            # If any direct neighbour is outside, this is an edge node
            if (mask[i - 1, j] or mask[i + 1, j] or
                mask[i, j - 1] or mask[i, j + 1]):
                outline_idx.append([i, j])

    outline_idx = np.asarray(outline_idx, dtype=int)
    outline_nodes = node[outline_idx[:, 0], outline_idx[:, 1], :]

    return outline_nodes[:, :2], outline_idx

def order_outer_contour(points):
    """
    Reorder unordered 2D outer contour points so they follow the boundary.

    points : (n,2) array of [x,y]

    returns ordered (n,2) array
    """
    pts = np.asarray(points)

    # centroid of the point cloud
    center = pts.mean(axis=0)

    # angle of each point relative to centroid
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])

    # sort points by angle
    order = np.argsort(angles)
    
    #center_point = (points.max(axis=0) + points.min(axis=0))/2
    
    return pts[order] # - center_point



# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

def validate_run_inputs(
    *,
    analysis_type,
    solver,
    tool_data=None,
    mesh=None,
    initial_conditions=None,
    analysis_options=None,
    layup=None,
    run_defaults=None,
):
    """
    Validate inputs for KinDrape3D run functions.

    Parameters
    ----------
    analysis_type : str
        Canonical analysis type identifier, for example:
        "single_layer", "straight_preform", "preshaped_preform", "layup".
    solver : str
        Canonical solver identifier, for example "NR" or "standard".
    tool_data : dict, optional
        Tool surface data produced by pre_process.tool_definition. 
    mesh : dict, optional
        Mesh definition. 
    initial_conditions : dict, optional
        Initial conditions. 
    analysis_options : dict, optional
        Analysis options. Some analysis types require keys such as
        "stabilizations".
    run_defaults : dict, optional
        Defaults bundle used for layup materialisation. Required when
        analysis_type is "layup".

    Returns
    -------
    None
        Raises ValueError if validation fails.
    """

    def require_dict(name, obj):
        if obj is None or not isinstance(obj, dict):
            raise ValueError(f'"{name}" must be a dict.')

    def require_keys(name, obj, keys):
        missing = [k for k in keys if k not in obj]
        if missing:
            raise ValueError(
                f'"{name}" is missing required key(s): {missing}.'
                )

    # --- Shared checks ------------------------------------------------------
    if initial_conditions is not None:
        if "pre_shear" not in initial_conditions:
            raise ValueError('"initial_conditions" must include "pre_shear".')

        pre_shear = initial_conditions["pre_shear"]
        if solver == "standard" and pre_shear != 0:
            raise ValueError(
                "Invalid configuration: pre_shear is only supported with the "
                'NR solver. Set analysis_options["solver"] to "NR" or set '
                'initial_conditions["pre_shear"] to 0.'
            )
            
    if solver == 'standard' and analysis_options['kinematics'] != 'pure_shear':
        raise ValueError(
                "Invalid configuration: the legacy standard solver is only "
                "supporter for pure shear analysis. "
            )

    # --- Analysis-type specific checks --------------------------------------
    if analysis_type == "single_layer":
        require_dict("tool_data", tool_data)
        require_keys("tool_data", tool_data, ["F"])

        require_dict("analysis_options", analysis_options)
        require_keys("analysis_options", analysis_options, 
                     ["solver", "kinematics", "stabilizations"])
    else:
        raise ValueError(f'Unknown analysis_type "{analysis_type}".')