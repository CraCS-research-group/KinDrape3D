"""
run.py

Main module for running the kinematic forming analyses.

This module handles the high-level input/output and calls the lower-level 
functions from the pre_process, kindrape_analysis, post_process, and utility 
modules.

Authors: Peter Hede Broberg, Christian Krogh, Bastian Schäfer, Jan Paul Wank, 
        Luise kärger, Brian Lau Verndal Bak, Esben Lindgaard
        
Date: February 2026
"""

import KinDrape3D as kd3d
import numpy as np


def run_kindrape3d(tool, mesh, initial_conditions, analysis_options, 
                   layup=None):
    """
    Select and run a KinDrape3D analysis.

    This is the main entry point for the analysis pipeline. It dispatches the
    user inputs to the relevant runner function.

    Parameters
    ----------
    tool : dict
        Tool definition. 
    mesh : dict
        Mesh and fabric definition. 
    initial_conditions : dict
        Draping start definition.
    analysis_options : dict
        Analysis configuration. It must include:
        - "analysis_type"
        - "solver"
        It may also include additional keys used by the selected analysis.
    layup : dict, optional
        Layup definition as a dictionary. This is required when
        `analysis_type` is "layup".

    Returns
    -------
    results : dict
        Results dictionary with a the contents:
        - "settings": the input dictionaries as provided to this function
        - "meta": analysis settings
        - "results": the unmodified output from the underlying runner function

        Note
        ----
        The contents of "results" vary by analysis type.
    """
    
    # --- Tool and analysis definition ---------------------------------------
    F, X, Y, Z = kd3d.pre_process.tool_definition(tool)
    tool_data = {"F": F, "X": X, "Y": Y, "Z": Z}

    analysis_type = analysis_options["analysis_type"]
    solver = analysis_options["solver"]

    # --- Validation of inputs -----------------------------------------------
    kd3d.utility.validate_run_inputs(
        analysis_type=analysis_type,
        solver=solver,
        tool_data=tool_data,
        mesh=mesh,
        initial_conditions=initial_conditions,
        analysis_options=analysis_options,
        layup=layup,
        run_defaults={
            "mesh": mesh,
            "initial_conditions": initial_conditions,
            "analysis_options": analysis_options,
        } if analysis_type == "layup" else None,
    )

    # --- Dispatch -----------------------------------------------------------
    if analysis_type == "single_layer":
        data = run_single_layer(tool_data, mesh, initial_conditions, 
                               analysis_options)
    else:
        # This should be unreachable because of the validation above.
        raise ValueError(f'Unhandled analysis_type "{analysis_type}".')
        
    # --- Package results dictionary -----------------------------------------
    return {
        "settings": {
            "tool": tool,
            "mesh": mesh,
            "initial_conditions": initial_conditions,
            "analysis_options": analysis_options,
            "layup": layup,
        },
        "meta": {
            "analysis_type": analysis_type,
            "solver": solver,
        },
        "sim_data": data,
        "tool_data": tool_data
    }

def run_single_layer(tool_data, mesh, initial_conditions, analysis_options):
    """
    Run a single-layer kinematic draping analysis on a tool surface.

    Parameters
    ----------
    tool_data : dict
        Tool surface data.
    mesh : dict
        Mesh definition.
    initial_conditions : dict
        Initial condition definition.
    analysis_options : dict
        Analysis configuration.

    Returns
    -------
    results : dict
        Results dictionary (data only). Keys:
        - "Node" : node array from the solver
        - "P" : cell corner point array
        - "Shear" : shear array
        - "e_22" : transverse strain array
        - "e_perp" : perpendicular strain array
        - "d" : cell spacing used by the solver
        - "Grid" : grid definition used by the solver
        - "masks" : dict with keys:
            - "node_outside_mask"
            - "cell_inside_mask_1d"
    """
    
    # --- Initialize variables -----------------------------------------------
    F = tool_data["F"]

    setup = kd3d.pre_process.setup_grid_and_initials(mesh, initial_conditions)
    Grid = setup["Grid"]
    d = setup["d"]
    OrgNode = setup["OrgNode"]
    Org = setup["Org"]
    Ang = setup["Ang"]
    masks = setup["masks"]

    # --- Run kinematic draping analysis -------------------------------------
    Shear, Node, P = kd3d.kindrape_analysis.kindrape_single_layer_combined(
        d, Grid, Org, Ang, OrgNode, F, analysis_options["solver"],
        initial_conditions["pre_shear"], analysis_options["kinematics"],
        analysis_options["stabilizations"],)
    
    e_22, e_perp = kd3d.post_process.compute_e22_and_perp_strain(P, d)
    
    # --- Apply trimming masks (orientation != 0 case) -----------------------
    cell_inside_mask_1d = masks["cell_inside_mask_1d"]
    if cell_inside_mask_1d is not None:
        Shear = Shear[cell_inside_mask_1d]
        e_22 = e_22[cell_inside_mask_1d]
        e_perp = e_perp[cell_inside_mask_1d]
        P = np.asarray(P)[cell_inside_mask_1d]

    return {
        "Node": Node,
        "P": P,
        "Shear": Shear,
        "e_22": e_22,
        "e_perp": e_perp,
        "d": d,
        "Grid": Grid,
        "masks": masks,
    }




