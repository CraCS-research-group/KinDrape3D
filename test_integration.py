"""
Integration tests for the KinDrape3D kinematic draping model.

Tests the complete analysis pipeline for the pure-shear, simple-shear, and 
enhanced simple-shear kinematic solver compred to baseline outputs.
"""
import numpy as np
import unittest
import KinDrape3D as kd3d
import warnings


class TestKinDrape3D(unittest.TestCase):

    ### Integration test 
    def test_full_integration_pure_shear(self):
        """
        Full integration test on the pure-shear solver

        Returns
        -------
        None.

        """
        warnings.filterwarnings("ignore") # To ignore warning regarding d 
        
        # Load the baseline results
        shear_baseline = kd3d.utility.load_results('test/shear_eff', True)
        node_baseline = kd3d.utility.load_results('test/node_eff', False)

        # Define variables
        x_tool, y_tool = np.meshgrid(np.linspace(0,2*np.pi,100),
                               np.linspace(1e-6,np.pi/2-np.pi/20,50))
        tool_parameters = {'radius':1.0}
        tool_type = 'hemisphere'
        tool = {'tool_type':tool_type, 'x_tool':x_tool, 'y_tool':y_tool, 
                'tool_parameters':tool_parameters}
        
        # Mesh
        mesh = {} # initialise the mesh dictionary
        mesh['d'] = 0.075
        mesh['L'] = 0.825
        mesh['W'] = 1.725
        mesh['orientation'] = 0
       
        # Initial conditions -- Starting point
        initial_conditions = {}
        initial_conditions['drape_start_mould_xy'] = [0,-0.9]
        initial_conditions['drape_start_fabric_us'] =  [0.075,0.075]
        initial_conditions['ang'] = 0
        initial_conditions['pre_shear'] = 20
        
        # Options
        analysis_options = {}
        analysis_options['analysis_type'] = 'single_layer'
        analysis_options['solver'] = 'NR' 
        analysis_options['kinematics'] = 'pure_shear'
        analysis_options['stabilizations'] = {}
        
        # Run analysis
        results = kd3d.run.run_kindrape3d(tool, mesh, initial_conditions, 
                                          analysis_options) 
        
        averaged_shear_results = np.mean(results['sim_data']['Shear'], 1)
        
        # Assert results
        tol= 1e-4
       
        assert np.allclose(averaged_shear_results, shear_baseline, 
                           atol=tol) # Shear
        assert np.allclose(results['sim_data']['Node'], node_baseline, 
                           atol=tol) # Nodes
        
    def test_full_integration_simple_shear(self):
        """
        Full integration test on the simple-shear kinematic solver

        Returns
        -------
        None.

        """
            
        # Load the baseline results
        shear_baseline = kd3d.utility.load_results('test/shear_simple_shear', 
                                                   True)
        node_baseline = kd3d.utility.load_results('test/node_simple_shear', 
                                                  False)

        # Define variables
        tool_type = 'point_cloud'
        tool = {
            'tool_type': tool_type,
            'file_name': 'Hemishpere_example\\hemisphere_xyz.XYZ',
            'centre': True
        }
            
        # Mesh
        mesh = {
            'd': 0.0125*2*12/16,
            'L': 0.28125,
            'W': 0.28125,
            'orientation': 0,
        }

        # Initial conditions -- Starting point
        initial_conditions = {
            'drape_start_mould_xy': [0, 0],
            'drape_start_fabric_us':[0.28125/2, 0.28125/2],
            'ang': 0,  # initial drape direction (deg)
            'pre_shear': 0  # initial in-plane shear (deg)
        }

        # Options
        analysis_options = {
            'analysis_type': 'single_layer',
            'solver': 'NR',
            'kinematics': 'simple_shear',
            'stabilizations': {
                'remove_spurious_points': True,
                'average_fiber_length': True,
                'curvature_based_point_removal': True,
                'use_irregularity_index': True,
                'curvature_threshold': 100,
                }
        }
            
        # Run analysis
        results = kd3d.run.run_kindrape3d(tool, mesh, initial_conditions, 
                                          analysis_options) 
           
        # Assert results
        tol= 1e-1
           
        assert np.allclose(results['sim_data']['Shear'], shear_baseline, 
                           atol=tol) # Shear
        assert np.allclose(results['sim_data']['Node'], node_baseline, 
                           atol=tol) # Nodes
            
    def test_full_integration_enhanced_simple_shear(self):
        """
        Full integration test on the enhanced simple-shear kinematic solver

        Returns
        -------
        None.

        """
                
        # Load the baseline results
        shear_baseline = kd3d.utility.load_results('test/shear_mod_simple_shear'
                                                   , True)
        node_baseline = kd3d.utility.load_results('test/node_mod_simple_shear'
                                                  , False)

        # Define variables
        tool_type = 'point_cloud'
        tool = {
            'tool_type': tool_type,
            'file_name': 'Hemishpere_example\\hemisphere_xyz.XYZ',
            'centre': True
        }
                
        # Mesh
        mesh = {
            'd': 0.0125*2*12/16,
            'L': 0.28125,
            'W': 0.28125,
            'orientation': 0,
        }

        # Initial conditions -- Starting point
        initial_conditions = {
            'drape_start_mould_xy': [+5.469446951953614e-17, 
                                     +5.469446951953614e-17],
            'drape_start_fabric_us':[0.28125/2, 0.28125/2], 
            'ang': 0,  # initial drape direction (deg)
            'pre_shear': 0  # initial in-plane shear (deg)
        }

        # Options
        analysis_options = {
            'analysis_type': 'single_layer',
                'solver': 'NR',
                'kinematics': 'enhanced_simple_shear',
                'stabilizations': {
                    'enhanced_simple_shear_relation_type': 'phenomenological',
                    'enhanced_simple_shear_parameters': [6, 0.6],
                    'enhanced_simple_shear_damping': 10,
                    'remove_spurious_points':True,
                    'average_fiber_length': True,
                    'curvature_based_point_removal': True,
                    'use_irregularity_index': True,
                    'curvature_threshold': 100,
                    }
            }
                
        # Run analysis
        results = kd3d.run.run_kindrape3d(tool, mesh, initial_conditions, 
                                          analysis_options)
                
        # Assert results
        tol= 1e-4
               
        assert np.allclose(results['sim_data']['Shear'], shear_baseline, 
                           atol=1e-1) # Shear
        assert np.allclose(results['sim_data']['Node'], node_baseline, 
                           atol=tol) # Nodes

if __name__ == '__main__':
    unittest.main()
    