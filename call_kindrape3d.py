"""
call_kindrape3d.py

Example script that shows how to set up input parameters and execute a 
kinematic draping simulation using the KinDrape3D module.

The script shows how to define:
- Tool geometry
- Mesh parameters
- Initial conditions
- Analysis options

Authors: Peter Hede Broberg, Christian Krogh, Bastian Schäfer, Jan Paul Wank, 
         Luise Kärger, Brian Lau Verndal Bak, Esben Lindgaard
"""

import time
import KinDrape3D as kd3d
import numpy as np

def main():
    # ---------------------------------------------------------------------
    # User choice: tool geometry case
    # ---------------------------------------------------------------------
    # Valid choices: "point_cloud", "hemisphere_fillet", "hemisphere", "ramp",
    #                "double_curved"
    tool_case = "hemisphere_fillet"
    # ---------------------------------------------------------------------
    # Tool definition
    # ---------------------------------------------------------------------
    if tool_case == "point_cloud":
        tool_type = "point_cloud"
        tool = {
            "tool_type": tool_type,
            "file_name": "Hemisphere_example/hemisphere_xyz.XYZ",
            "centre": True,
        }

    elif tool_case == "hemisphere_fillet":
        tool_type = "hemisphere_fillet"
        x_tool, y_tool = np.meshgrid(
            np.linspace(-0.2, 0.2, 100),
            np.linspace(-0.2, 0.2, 100),
        )
        tool_parameters = {"R": 0.075, "r_f": 0.020}
        tool = {
            "tool_type": tool_type,
            "x_tool": x_tool,
            "y_tool": y_tool,
            "tool_parameters": tool_parameters,
        }
    
    elif tool_case == "hemisphere":
        tool_type = "hemisphere"
        x_tool, y_tool = np.meshgrid(
            np.linspace(0, 2 * np.pi, 100),
            np.linspace(1e-6, np.pi / 2 - np.pi / 20, 50),
        )
        tool_parameters = {"radius": 0.2}
        tool = {
            "tool_type": tool_type,
            "x_tool": x_tool,
            "y_tool": y_tool,
            "tool_parameters": tool_parameters,
        }
    
    elif tool_case == "ramp":
        tool_type = "ramp"
        x_tool, y_tool = np.meshgrid(
            np.linspace(0, 1.1, 500),
            np.linspace(0, 0.2, 100),
        )
        tool_parameters = {
            "l_flanges": 0.3,
            "h_ramp": 0.15,
            "l_ramp": 0.32,
            "r_transition": 0.05,
        }
        tool = {
            "tool_type": tool_type,
            "x_tool": x_tool,
            "y_tool": y_tool,
            "tool_parameters": tool_parameters,
        }
    
    elif tool_case == "double_curved":
        tool_type = "double_curved"
        x_tool, y_tool = np.meshgrid(
            np.linspace(0.0, 0.5, 51),
            np.linspace(0.0, 0.5, 51),
        )
        tool_parameters = {}
        tool = {
            "tool_type": tool_type,
            "x_tool": x_tool,
            "y_tool": y_tool,
            "tool_parameters": tool_parameters,
        }
    
    else:
        raise ValueError(f"Unknown tool_case: {tool_case}")

    # ---------------------------------------------------------------------
    # Mesh definition
    # ---------------------------------------------------------------------
    mesh = {
        # Length of the fabric
        "L": 0.3, 
        
        # Width of the fabric
        "W": 0.3, 
        
        # Orientation of the fibres along the warp direction
        "orientation": 0.0, 
        
        # Cell spacing discretization
        "d": 0.00492,   
    }

    # ---------------------------------------------------------------------
    # Initial conditions
    # ---------------------------------------------------------------------
    initial_conditions = {
        # Origin point on the mould
        "drape_start_mould_xy": [0.0, 0.0], 
        
        # Origin point on the fabric
        "drape_start_fabric_us": [mesh["L"] / 2, mesh["W"] / 2], 
        
        # Draping direction
        "ang": 0.0,
        
        # Pre-shear in the fabric. Only valid for pure shear analysis
        "pre_shear": 0.0, 
    }
    
    # ---------------------------------------------------------------------
    # Analysis options
    # ---------------------------------------------------------------------
    analysis_options = {
        # Analysis type
        # Valid options: 'single_layer'
        "analysis_type": "single_layer",  
        
        # Solver type
        # Valid otions: "NR" or "standard". 
        # Note: "standard" is a legacy solver only valid for pure shear
        "solver": "NR",  
        
        # Kinematic assumptions
        # Valid options: 'pure_shear', 'simple_shear', 'enhanced_simple_shear'
        "kinematics": "enhanced_simple_shear", 
        
        # Stabilizations for the simple shear and enhanced simple shear
        "stabilizations": {
            # Options: phenomenological, tricot, tricot_bounded, simple_shear
            "enhanced_simple_shear_relation_type": "phenomenological", 
            "enhanced_simple_shear_parameters":  [4.313,  4.709e-01],
            "enhanced_simple_shear_damping": 10,
            "remove_spurious_points": True,
            "average_fiber_length": True,
            "curvature_based_point_removal": True,
            # Only applicable if curvature_based_point_removal is True
            "use_irregularity_index": True, 
            "curvature_threshold": 300, 
        },
    }

    # ---------------------------------------------------------------------
    # Run draping simulation
    # ---------------------------------------------------------------------
    tic = time.perf_counter()
    results = kd3d.run.run_kindrape3d(tool, mesh, initial_conditions, 
                                      analysis_options)
    toc = time.perf_counter()

    print(f"Time spent on draping analysis: {toc - tic:.2f} s")
    
    # ---------------------------------------------------------------------
    # Example of plotting and saving single layer draping results
    # ---------------------------------------------------------------------
    
    # Plotting the results
    P = results['sim_data']['P']
    X = results['tool_data']['X']
    Y = results['tool_data']['Y']
    Z = results['tool_data']['Z']
    
    ax, fig = kd3d.utility.plot_drape_results(
        results['sim_data']['e_perp'], 
        P, X, Y, Z, 
        label="Perpendicular strain [-]")
    
    ax, fig = kd3d.utility.plot_drape_results(
        results['sim_data']['Shear'], 
        P, X, Y, Z, 
        label="In-plane shear [deg]")
    
    # Retrieve the outer contour
    nodes = results['sim_data']['Node']
    if mesh['orientation'] == 0: 
        outer_contour = kd3d.utility.find_outer_contour(nodes)
    else:
        mask = results['sim_data']['masks']['node_outside_mask']
        outer_contour, _ = kd3d.utility.find_masked_outer_contour(nodes, mask) 
        outer_contour = kd3d.utility.order_outer_contour(outer_contour)
    
    # Saving the results
    kd3d.utility.save_drape_results(
        results['sim_data'], results['settings'], toc-tic, fig, ax, 
        "example_id", "run_id", save_figures=True,
        )
    # Loading results
    # run_dir, fields, meta = kd3d.utility.load_drape_results('example_id', 
    #                                                         'run_id')
    return results

if __name__ == '__main__':
    main()