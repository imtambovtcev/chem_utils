
# Chem Utils

Chem Utils is a Python package for representing and manipulating molecular structures, motors, fragments, and electron densities, among other functionalities.

## Installation

### 1. Clone the repository to your local machine:
```bash
git clone https://gitlab.com/imtambovtcev/chem_utils.git
```

### 2. Navigate to the cloned repository:
```bash
cd chem_utils
```

### 3. Install the package in editable mode:
```bash
pip install -e .
```

If you need optional dependencies such as `PyQt5` for GUI-related features, install the package with the `qt` extras:

```bash
poetry install --extras "qt"
```

## Modules

### 1. `molecule.py`
Represents and manipulates molecular structures. Provides functionality for loading molecules, manipulating coordinates, bonds, and saving them in various formats.

### 2. `motor.py`
Represents a Motor, extending the functionality of the `Molecule` class. Used for motor-like molecular systems with specialized handling of movement and conformation.

### 3. `fragment.py`
Represents a molecular fragment, extending the functionality of the `Molecule` class. Useful for handling parts or fragments of larger molecular systems.

### 4. `eldens.py`
Deals with the representation and manipulation of electron density data, including functions for loading `.cube` files and visualizing electron densities.

### 5. `path.py`
Represents a sequence or collection of molecular configurations (images), useful for handling transition pathways between molecular states.

### 6. `frequency.py`
Represents and manipulates vibrational frequencies and modes, enabling analysis of molecular vibrations and related properties.

### 7. `utils.py`
Provides various utility functions used throughout the package, including functions for common operations in molecular manipulations.

### 8. `xyz_from_allxyz.py`
Contains functions to extract XYZ coordinates from a file containing multiple XYZ coordinate sets. Useful for handling large sets of molecular data.

### 9. `neb_plot.py`
Generates plots for Nudged Elastic Band (NEB) calculations, helping visualize the energy landscape and transition paths.

### 10. `get_image.py`
Helps retrieve images associated with molecules or visual representations of molecular configurations.

## Usage

Each module can be used individually or together to build complex workflows for molecular modeling and analysis. The package also comes with a number of command-line scripts (see below) for ease of use in different scenarios.

## CLI Scripts

The package provides several command-line scripts, which can be run after installation. Here’s a list of the available scripts:

- `neb_plot`: Generates plots for NEB calculations.
- `xyz_to_allxyz`: Converts single XYZ files to a multi-structure `.allxyz` file.
- `check_and_copy_path`: Checks a molecular path and copies files if needed.
- `format_xyz`: Formats XYZ files for easy reading or further processing.
- `get_image`: Retrieves images associated with molecules.
- `get_orb`: Extracts molecular orbital information from quantum chemistry outputs.
- `mirror_xyz`: Mirrors the coordinates in an XYZ file along a specified axis.
- `molecular_distance`: Calculates distances between molecular structures.
- `render_molecule`: Renders a 3D image of a molecule using PyVista.
- `rotate_path`: Rotates a sequence of molecular configurations.
- `allxyz_to_xyz`: Converts `.allxyz` files back into individual XYZ files.
- `plot_spectrum`: Plots vibrational or electronic spectra from molecular calculations.
- `geometry_convergence_plot`: Plots the geometry convergence during molecular optimizations.
- `shake_molecule`: Perturbs molecular structures by shaking atomic positions randomly.

## Contributing

To contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Submit a pull request for review.

For more detailed contributing guidelines, please refer to the `CONTRIBUTING.md` file.