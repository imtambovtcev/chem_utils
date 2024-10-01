import re

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

from .constants import BOHR_TO_ANGSTROM

DEFAULT_ELDENS_SETTINGS = {'show': True, 'isosurface_value': 0.1,
                           'show_grid_surface': False, 'show_grid_points': False}


class ElectronDensity:
    def __init__(self, electron_density, org, xvec, yvec, zvec):
        self.electron_density = electron_density
        self.org = np.array(org)
        self.xvec = np.array(xvec)
        self.yvec = np.array(yvec)
        self.zvec = np.array(zvec)

        # Derive points and dimensions from electron_density
        nx, ny, nz = self.electron_density.shape
        self.dimensions = [nx, ny, nz]

    @property
    def points(self):
        nx, ny, nz = self.dimensions

        x_indices = np.linspace(0, nx - 1, nx)[:, None, None, None]
        y_indices = np.linspace(0, ny - 1, ny)[None, :, None, None]
        z_indices = np.linspace(0, nz - 1, nz)[None, None, :, None]

        # For each point in the grid, compute its coordinates by adding the contributions of the xvec, yvec, zvec, and the origin
        points = (self.org[None, None, None, :]
                  + x_indices * self.xvec[None, None, None, :]
                  + y_indices * self.yvec[None, None, None, :]
                  + z_indices * self.zvec[None, None, None, :])

        points = points.reshape(nx, ny, nz, 3)

        points = np.swapaxes(points, 0, 2)

        return points

    @staticmethod
    def _getline(cube):
        """
        Reads a line from the cube file and parses it into appropriate types.
        Returns a tuple where the first element is an integer (natm or grid points),
        and the second element is a list of floats (origin or vector components).
        """
        parts = cube.readline().strip().split()
        # Ensure there are parts to parse
        if not parts:
            raise ValueError(
                "Unexpected end of file or empty line encountered.")
        # Try to parse the first part as an integer
        try:
            first_value = int(parts[0])
            rest_values = [float(x) for x in parts[1:]]
            return first_value, rest_values
        except ValueError:
            # If parsing fails, raise an error
            raise ValueError(
                f"Expected an integer in the first column, got '{parts[0]}'.")

    @staticmethod
    def read_cube(fname, cube_format='ORCA', vector_permutation=None, axis_permutation=None, coordinate_permutation=None, unit_conversion=None):
        """
        Reads a cube file and extracts metadata and volumetric data.

        Parameters:
            fname (str): Path to the cube file.
            cube_format (str): 'ORCA' or 'GPAW', specifying the format of the cube file. Default is 'ORCA'.
            vector_permutation (tuple): A tuple specifying the permutation of grid vectors and dimensions.
                                        For example, (0, 1, 2) means no change, (1, 0, 2) swaps the first two.
            axis_permutation (tuple): A tuple specifying the permutation of axes to apply to the data array.
                                    For example, (0, 1, 2) means no change, (1, 2, 0) rearranges axes.
            coordinate_permutation (tuple): A tuple specifying the permutation of the coordinate axes.
                                            For example, (0, 1, 2) means no change, (1, 0, 2) swaps x and y coordinates.

        Returns:
            numpy.ndarray: A 3D array containing the volumetric data.
            dict: A dictionary containing metadata extracted from the cube file.
        """
        CUBE_FORMAT_PRESETS = {
            'ORCA': {
                'unit_conversion': True,
                'vector_permutation': None,
                'axis_permutation': None,
                'coordinate_permutation': None,
            },
            'GPAW': {
                'unit_conversion': True,
                'vector_permutation': [0, 1, 2],
                'axis_permutation': [0, 1, 2],
                'coordinate_permutation': [2, 1, 0],
            },
        }

        # Apply presets based on cube_format
        presets = CUBE_FORMAT_PRESETS.get(cube_format.upper(), {})
        if unit_conversion is None:
            unit_conversion = presets.get('unit_conversion', True)
        if vector_permutation is None:
            vector_permutation = presets.get('vector_permutation', None)
        if axis_permutation is None:
            axis_permutation = presets.get('axis_permutation', None)
        if coordinate_permutation is None:
            coordinate_permutation = presets.get(
                'coordinate_permutation', None)

        meta = {}
        with open(fname, 'r') as cube:
            # Read the first two comment lines in the cube file
            comment1 = cube.readline().strip()
            comment2 = cube.readline().strip()

            # Default loop order
            loop_order = ['z', 'y', 'x']

            if 'OUTER LOOP' in comment2:
                # Parse the loop order
                loop_order_line = comment2
                # Example: 'OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z'
                match = re.search(
                    r'OUTER LOOP:\s*(\w+),\s*MIDDLE LOOP:\s*(\w+),\s*INNER LOOP:\s*(\w+)', loop_order_line)
                if match:
                    # Loop order from outer to inner
                    loop_order = [match.group(3).lower(), match.group(
                        2).lower(), match.group(1).lower()]
                else:
                    # If parsing fails, use default
                    pass

            # Read metadata: number of atoms (natm) and origin (meta['org'])
            natm, meta['org'] = ElectronDensity._getline(cube)

            # Read the number of points and vector information in each dimension
            grid_info = [ElectronDensity._getline(cube) for _ in range(3)]
            nums = [n for n, vec in grid_info]
            vecs = [vec for n, vec in grid_info]

            # Units handling
            if unit_conversion:
                # Convert from Bohr to Angstroms for origin and vectors
                meta['org'] = [x * BOHR_TO_ANGSTROM for x in meta['org']]
                vecs = [[x * BOHR_TO_ANGSTROM for x in vec] for vec in vecs]

            # Apply vector_permutation to nums and vecs
            if vector_permutation is not None:
                nums = [nums[i] for i in vector_permutation]
                vecs = [vecs[i] for i in vector_permutation]
            else:
                # Default mapping: nums[0] -> nx, vecs[0] -> xvec, etc.
                pass

            # Assign nums and vecs to meta
            nx, ny, nz = nums
            meta['xvec'], meta['yvec'], meta['zvec'] = vecs

            # Extract atom information, considering the absolute value of natm
            natm_abs = abs(natm)
            meta['atoms'] = [ElectronDensity._getline(
                cube) for _ in range(natm_abs)]

            data_values = []

            # In ORCA cube files, there may be an extra line before data that should be skipped
            if cube_format == 'ORCA':
                # Read the next line after the atom lines
                line = cube.readline()
                values_in_line = [float(val) for val in line.strip().split()]
                if len(values_in_line) == 2:
                    pass  # Skip this line
                else:
                    data_values.extend(values_in_line)
            # Now read the data
            for line in cube:
                values_in_line = [float(val) for val in line.strip().split()]
                data_values.extend(values_in_line)

            # Check if the read data points match the expected data points
            expected_data_points = nx * ny * nz
            if len(data_values) != expected_data_points:
                raise ValueError(
                    f"Number of data points in the file ({len(data_values)}) does not match the expected size ({expected_data_points})")

            # Map axis names to dimensions
            axes_to_dims = {'x': nx, 'y': ny, 'z': nz}
            # Get the dimensions in the order of loop_order
            dims = [axes_to_dims[axis] for axis in loop_order]
            # Reshape the data_values into the dimensions in loop_order
            data = np.array(data_values).reshape(dims)

            # Now we need to rearrange axes to get data in order [x, y, z]
            # If axis_permutation is provided, use it
            if axis_permutation is not None:
                data = data.transpose(axis_permutation)
            else:
                # Default to rearranging axes to get [x, y, z]
                current_axes = loop_order  # axes in data after reshaping
                desired_axes = ['x', 'y', 'z']
                axis_permutation = [current_axes.index(
                    axis) for axis in desired_axes]
                data = data.transpose(axis_permutation)

            # Build the coordinate grid using the origin and grid vectors
            idx = [np.arange(n) for n in (nx, ny, nz)]
            grid = np.meshgrid(*idx, indexing='ij')

            # Compute the points in space
            points = np.zeros((nx, ny, nz, 3))
            for i, vec in enumerate([meta['xvec'], meta['yvec'], meta['zvec']]):
                points += grid[i][..., None] * np.array(vec)

            points += np.array(meta['org'])

            # If coordinate_permutation is provided, apply it to the points
            if coordinate_permutation is not None:
                points = points[..., coordinate_permutation]
                # Also permute the grid vectors and origin accordingly
                meta['xvec'], meta['yvec'], meta['zvec'] = [meta[vec]
                                                            for vec in ['xvec', 'yvec', 'zvec']]
                vecs = [meta['xvec'], meta['yvec'], meta['zvec']]
                vecs = [vecs[i] for i in coordinate_permutation]
                meta['xvec'], meta['yvec'], meta['zvec'] = vecs
                meta['org'] = [meta['org'][i] for i in coordinate_permutation]

            # Store points in meta for further use
            meta['points'] = points

        return data, meta

    @classmethod
    def load(cls, cube_file_path, cube_format='ORCA', vector_permutation=None, axis_permutation=None, coordinate_permutation=None):
        # Read cube file and extract data and essential metadata
        data, meta = cls.read_cube(
            cube_file_path, cube_format=cube_format, vector_permutation=vector_permutation, axis_permutation=axis_permutation, coordinate_permutation=coordinate_permutation)

        # Return an instance of ElectronDensity initialized with data and essential metadata
        return cls(data, meta['org'], meta['xvec'], meta['yvec'], meta['zvec'])

    def copy(self):
        """
        Creates a copy of the ElectronDensity instance.

        Returns:
            ElectronDensity: A new instance of ElectronDensity with the same attributes as the original.
        """
        # Create a new instance of ElectronDensity with the same attributes as the original instance
        return ElectronDensity(np.copy(self.electron_density),
                               np.copy(self.org),
                               np.copy(self.xvec),
                               np.copy(self.yvec),
                               np.copy(self.zvec))

    def rotate(self, rotation_matrix):
        assert rotation_matrix.shape == (
            3, 3), "Rotation matrix must be a 3x3 matrix."

        assert np.allclose(rotation_matrix.T, np.linalg.inv(
            rotation_matrix)), "Rotation matrix is not orthogonal"
        assert np.isclose(np.linalg.det(rotation_matrix),
                          1), "Determinant of rotation matrix is not 1"

        self.xvec = np.dot(rotation_matrix, self.xvec)
        self.yvec = np.dot(rotation_matrix, self.yvec)
        self.zvec = np.dot(rotation_matrix, self.zvec)
        self.org = np.dot(rotation_matrix, self.org)

    def translate(self, translation_vector):
        """
        Translates the points by the given translation vector.

        :param translation_vector: 1x3 translation vector
        """
        assert len(
            translation_vector) == 3, "Translation vector must be a 1x3 vector."

        # Translate the origin
        self.org += translation_vector

    def resample_to(self, target, method='nearest'):

        # Get the points and values from the 'self' instance
        # Coordinates of the points in the 'self' grid
        points_self = self.points.reshape(-1, 3)
        # Electron density values at the points in the 'self' grid
        values_self = self.electron_density.ravel()

        # Get the points from the 'target' instance
        # Coordinates of the points in the 'target' grid
        points_target = target.points.reshape(-1, 3)

        # Interpolate the values from the 'self' grid to the 'target' grid using scipy's griddata
        values_target = griddata(
            points_self, values_self, points_target, method=method, fill_value=0.0)

        # Reshape the interpolated values to match the shape of the target's electron_density
        values_target = values_target.reshape(target.electron_density.shape)

        # Return a new ElectronDensity instance with the interpolated values and the meta of the 'target' instance
        return ElectronDensity(values_target, target.org, target.xvec, target.yvec, target.zvec)

    def subtract(self, other, method='nearest'):
        """
        Subtracts the electron density of another ElectronDensity instance from this instance.

        :param other: Another ElectronDensity instance
        :return: A new ElectronDensity instance representing the difference
        """
        # Check if other is an instance of ElectronDensity
        if not isinstance(other, ElectronDensity):
            raise TypeError(
                "The 'other' parameter must be an instance of ElectronDensity.")

        # Resample 'other' instance to 'self' instance
        other_resampled = other.resample_to(self, method=method)

        # Calculate the difference of the electron densities and create a new instance
        difference = self.electron_density - other_resampled.electron_density
        return ElectronDensity(difference, self.org, self.xvec, self.yvec, self.zvec)

    def render(self, plotter=None, isosurface_value=0.1, isosurface_color='b', show_grid_surface=False,
               show_grid_points=False, notebook=False, opacity=0.3, grid_surface_color="b",
               grid_points_color="r", grid_points_size=5, save=None, show=False, smooth_surface=True):

        if plotter is None:
            if save:
                plotter = pv.Plotter(notebook=False, off_screen=True,
                                     line_smoothing=True, polygon_smoothing=True, image_scale=5)
            else:
                plotter = pv.Plotter(notebook=notebook)

        nx, ny, nz = self.dimensions
        x, y, z = self.points.T.reshape(3, nx, ny, nz)
        grid = pv.StructuredGrid(x, y, z)

        if isosurface_value is not None:
            contour = grid.contour(
                scalars=self.electron_density.ravel(), isosurfaces=[isosurface_value])
            if smooth_surface:
                contour = contour.subdivide(nsub=2, subfilter='loop')
                contour = contour.smooth(n_iter=50)

            plotter.add_mesh(contour, color=isosurface_color,
                             opacity=opacity, show_scalar_bar=False)

        if show_grid_surface:
            plotter.add_mesh(grid.outline(), color=grid_surface_color)

        if show_grid_points:
            plotter.add_mesh(grid, style='points', point_size=grid_points_size,
                             color=grid_points_color, render_points_as_spheres=True)

        # If saving is required, save the screenshot
        if isinstance(save, str):
            plotter.show(window_size=[1000, 1000])
            plotter.screenshot(save)

        # If showing is required, display the visualization
        if show:
            plotter.show(window_size=[1000, 1000], interactive=True)

        return plotter

    def __str__(self):
        info = []
        info.append(f"ElectronDensity Instance:")
        info.append(f"Dimensions: {self.dimensions}")
        info.append(f"Number of Points: {self.points.shape[0]}")

        # Extracting some information
        info.append(f"Org: {self.org}")
        info.append(f"Xvec: {self.xvec}")
        info.append(f"Yvec: {self.yvec}")
        info.append(f"Zvec: {self.zvec}")

        # Find the maximum value of electron density and its coordinates
        max_density_idx = np.argmax(self.electron_density)
        max_density_value = self.electron_density.ravel()[max_density_idx]
        max_density_coords = np.unravel_index(
            max_density_idx, self.electron_density.shape)
        max_density_point = self.points.reshape(
            *self.dimensions, 3)[max_density_coords]

        info.append(f"Maximum Electron Density Value: {max_density_value}")
        info.append(
            f"Coordinates of Maximum Electron Density: {max_density_point}")

        return '\n'.join(info)
