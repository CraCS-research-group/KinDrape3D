# KinDrape3D

KinDrape3D is a Python code for in-plane kinematic draping analysis of composite textile reinforcements over mould surfaces.

This repository contains the in-plane kinematic model used in: 
P.H. Broberg, C. Krogh, B. Schäfer, J.P. Wank, L. Kärger, B.L.V. Bak, E. Lindgaard. An enhanced kinematic forming framework for fast and accurate prediction of in-plane strains in unidirectional fabrics. In review. 

It supports kinematic mapping of a fabric grid onto a tool surface and computes deformation measures such as in-plane shear, transverse strain, and perpendicular strain for single-layer draping analyses.

The current release includes the simple-shear and enhanced simple-shear algorithms introduced in the associated paper. Out-of-plane kinematic extensions will be added in a later release.

## Installation

KinDrape3D requires Python 3.11 or newer and the following Python packages:

```bash
pip install numpy scipy matplotlib
```

For development and testing, install `pytest` as well:

```bash
pip install pytest
```

Alternatively, the required packages can be installed in a Conda environment:

```bash
conda install numpy scipy matplotlib pytest
```

## Basic usage

Most analyses follow the same workflow:

1. Define the mould surface, either as a parametric surface or from point-cloud data.
2. Define the fabric grid, including fabric dimensions, spacing, and orientation.
3. Define the initial draping conditions.
4. Select analysis options, including solver type and kinematic assumptions.
5. Run the analysis with `run_kindrape3d`.
6. Post-process, plot, and save the results.

A complete example is provided in:

```text
call_kindrape3d.py
```

The main analysis function is called as:

```python
run_kindrape3d(tool, mesh, initial_conditions, analysis_options)
```

## Repository structure

```text
run.py
    Orchestrates the simulation by calling the pre-processor, solver, and post-processor.

pre_process.py
    Prepares the tool surface, converts fabric dimensions to grids,
    and computes start conditions and masks.

kindrape_analysis.py
    Contains the kinematic draping solvers.

post_process.py
    Computes derived deformation measures.

utility.py
    Contains plotting, visualisation, save/load functions, and helper routines.

call_kindrape3d.py
    Provides an example setup for defining and running an analysis.
```

## Testing

Run the test with:

```bash
pytest
```

## Documentation

The source files contain docstrings and comments that describe the main functions, input variables, and outputs. The associated journal paper provides the full theoretical background and validation.

## Citation

If you use this code in your research, please cite:

```text
P.H. Broberg, C. Krogh, B. Schäfer, J.P. Wank, L. Kärger, B.L.V. Bak, E. Lindgaard. An enhanced kinematic forming framework for fast and accurate prediction of in-plane strains in unidirectional fabrics. In review. 
```
