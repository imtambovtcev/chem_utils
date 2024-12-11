import re

import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator, griddata

from .constants import BOHR_TO_ANGSTROM

DEFAULT_SCALAR_FIELD_SETTINGS = {'show': True, 'isosurface_value': 0.1,
                                 'show_grid_surface': False, 'show_grid_points': False}


class ScalarField:
    def __init__(self, scalar_field, org, lat1, lat2, lat3):
        """
         self.scalar_field: 3D numpy array representing the scalar field, outermost dimension corresponds to x-axis, middle dimension corresponds to y-axis, and innermost dimension corresponds to z-axis if loaded from standard cube file.
        """

        self.scalar_field = scalar_field
        self.org = np.array(org)

        self.lat1 = np.array(lat1)
        self.lat2 = np.array(lat2)
        self.lat3 = np.array(lat3)

        # Derive points and dimensions from scalar_field
        n1, n2, n3 = self.scalar_field.shape
        self.dimensions = [n1, n2, n3]

    @property
    def points(self):
        # Generate grid indices for each dimension
        indices = np.indices(self.dimensions)
        i, j, k = indices[0], indices[1], indices[2]

        # Compute coordinates for each point in the scalar field
        points = (
            self.org +
            i[..., np.newaxis] * self.lat1 +
            j[..., np.newaxis] * self.lat2 +
            k[..., np.newaxis] * self.lat3
        )
        return points

    @property
    def volume_element(self):
        """
        Computes the volume element, which is the volume represented by each grid cell.

        Returns:
            float: The volume of a single grid cell in the scalar field.
        """
        # Calculate the volume element as the scalar triple product of the grid vectors
        assert abs(abs(self.lat1[0])-np.linalg.norm(self.lat1)
                   ) < 1e-6, "lat1 is not parallel to the x-axis"
        assert abs(abs(self.lat2[1])-np.linalg.norm(self.lat2)
                   < 1e-6), "lat2 is not parallel to the y-axis"
        assert abs(abs(self.lat3[2])-np.linalg.norm(self.lat3)
                   < 1e-6), "lat3 is not parallel to the z-axis"
        return np.array([self.lat1[0], self.lat2[1], self.lat3[2]])

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
    def read_cube(fname, unit_conversion=True):
        """
        Reads a cube file and extracts metadata and volumetric data.
        """
        meta = {}
        with open(fname, 'r') as cube:
            # Read the first two comment lines in the cube file
            comment1 = cube.readline().strip()
            comment2 = cube.readline().strip()

            # Default loop order is now 'xyz'
            loop_order = ['x', 'y', 'z']

            # Read metadata: number of atoms (natm) and origin (meta['org'])
            natm, meta['org'] = ScalarField._getline(cube)

            # Read the number of points and vector information in each dimension
            grid_info = [ScalarField._getline(cube) for _ in range(3)]
            nums = [n for n, vec in grid_info]
            vecs = [vec for n, vec in grid_info]

            # Assign nums and vecs to meta
            n1, n2, n3 = nums

            # Extract atom information, considering the absolute value of natm
            natm_abs = abs(natm)

            def format_atom(atom):
                return (atom[0], np.array(atom[1])[1:])
            meta['atoms'] = [format_atom(ScalarField._getline(
                cube)) for _ in range(natm_abs)]

            # Units handling
            if unit_conversion:
                # Convert from Bohr to Angstroms for origin and vectors
                def convert_atom(atom, factor):
                    return (atom[0], [x * factor for x in atom[1]])
                meta['atoms'] = [convert_atom(
                    atom, BOHR_TO_ANGSTROM) for atom in meta['atoms']]
                meta['org'] = [x * BOHR_TO_ANGSTROM for x in meta['org']]
                vecs = [[x * BOHR_TO_ANGSTROM for x in vec] for vec in vecs]

            meta['lat1'], meta['lat2'], meta['lat3'] = vecs

            data_values = []

            # Skip extra line for ORCA files

            line = cube.readline()
            values_in_line = [float(val) for val in line.strip().split()]
            if len(values_in_line) == 2:
                pass
            else:
                data_values.extend(values_in_line)

            for line in cube:
                values_in_line = [float(val) for val in line.strip().split()]
                data_values.extend(values_in_line)

            # Check if the read data points match the expected data points
            expected_data_points = n1 * n2 * n3
            if len(data_values) != expected_data_points:
                raise ValueError(
                    f"Number of data points in the file ({len(data_values)}) does not match the expected size ({expected_data_points})")

            # Map axis names to dimensions
            axes_to_dims = {'x': n1, 'y': n2, 'z': n3}
            dims = [axes_to_dims[axis] for axis in loop_order]
            data = np.array(data_values).reshape(dims)

        return data, meta

    @classmethod
    def load_cube(cls, cube_file_path):
        # Read cube file and extract data and essential metadata
        data, meta = cls.read_cube(cube_file_path)

        # Return an instance of ElectronDensity initialized with data and essential metadata
        return cls(data, meta['org'], meta['lat1'], meta['lat2'], meta['lat3'])

    def copy(self):
        """
        Creates a copy of the ElectronDensity instance.

        Returns:
            ElectronDensity: A new instance of ElectronDensity with the same attributes as the original.
        """
        # Create a new instance of ElectronDensity with the same attributes as the original instance
        return ScalarField(np.copy(self.scalar_field),
                           np.copy(self.org),
                           np.copy(self.lat3),
                           np.copy(self.lat2),
                           np.copy(self.lat1))

    def rotate(self, rotation_matrix):
        assert rotation_matrix.shape == (
            3, 3), "Rotation matrix must be a 3x3 matrix."

        assert np.allclose(rotation_matrix.T, np.linalg.inv(
            rotation_matrix)), "Rotation matrix is not orthogonal"
        assert np.isclose(np.linalg.det(rotation_matrix),
                          1), "Determinant of rotation matrix is not 1"

        self.lat3 = np.dot(rotation_matrix, self.lat3)
        self.lat2 = np.dot(rotation_matrix, self.lat2)
        self.lat1 = np.dot(rotation_matrix, self.lat1)
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
        # scalar field values at the points in the 'self' grid
        values_self = self.scalar_field.ravel()

        # Get the points from the 'target' instance
        # Coordinates of the points in the 'target' grid
        points_target = target.points.reshape(-1, 3)

        # Interpolate the values from the 'self' grid to the 'target' grid using scipy's griddata
        values_target = griddata(
            points_self, values_self, points_target, method=method, fill_value=0.0)

        # Reshape the interpolated values to match the shape of the target's scalar_field
        values_target = values_target.reshape(target.scalar_field.shape)

        # Return a new ElectronDensity instance with the interpolated values and the meta of the 'target' instance
        return ScalarField(values_target, target.org, target.xvec, target.yvec, target.zvec)

    def subtract(self, other, method='nearest'):
        """
        Subtracts the scalar field of another ElectronDensity instance from this instance.

        :param other: Another ElectronDensity instance
        :return: A new ElectronDensity instance representing the difference
        """
        # Check if other is an instance of ElectronDensity
        if not isinstance(other, ScalarField):
            raise TypeError(
                "The 'other' parameter must be an instance of ElectronDensity.")

        # Resample 'other' instance to 'self' instance
        other_resampled = other.resample_to(self, method=method)

        # Calculate the difference of the scalar fields and create a new instance
        difference = self.scalar_field - other_resampled.scalar_field
        return ScalarField(difference, self.org, self.lat3, self.lat2, self.lat1)

    def scalar_field_along_line(self, start_point, end_point, num_points=100) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts scalar field values along a line between two points in space.

        Parameters:
            start_point (array-like): Starting point of the line in 3D space.
            end_point (array-like): Ending point of the line in 3D space.
            num_points (int): Number of points along the line at which to sample the scalar field.

        Returns:
            tuple: A tuple containing:
                - line_points: The sampled points along the line.
                - line_values: The scalar field values at each of these points.
        """

        # Ensure points are in the correct format
        start_point = np.array(start_point)
        end_point = np.array(end_point)

        # Generate the points along the line
        line_points = np.linspace(start_point, end_point, num_points)

        # Prepare grid data for interpolation
        n1, n2, n3 = self.dimensions
        x = np.linspace(0, n1 - 1, n1)
        y = np.linspace(0, n2 - 1, n2)
        z = np.linspace(0, n3 - 1, n3)

        # Create the interpolator
        interpolator = RegularGridInterpolator(
            (x, y, z), self.scalar_field, bounds_error=False, fill_value=0)

        # Transform line_points to the scalar field's coordinate space
        transformed_points = (
            line_points - self.org) @ np.linalg.inv(np.array([self.lat3, self.lat2, self.lat1]))

        # Sample the scalar field along the transformed line points
        line_values = interpolator(transformed_points)

        return line_points, line_values

    def render(self, plotter=None, isosurface_value=0.1, isosurface_color='b', show_grid_surface=False,
               show_grid_points=False, notebook=False, opacity=0.3, grid_surface_color="b",
               grid_points_color="r", grid_points_size=5, save=None, show=False, smooth_surface=True,
               show_filtered_points=False, point_value_range=(0.0, 1.0)):

        # Initialize plotter if not provided
        if plotter is None:
            if save:
                plotter = pv.Plotter(notebook=False, off_screen=True,
                                     line_smoothing=True, polygon_smoothing=True, image_scale=5)
            else:
                plotter = pv.Plotter(notebook=notebook)

        # Extract lattice coordinates from points
        n1, n2, n3 = self.dimensions
        coord1, coord2, coord3 = self.points[...,
                                             0], self.points[..., 1], self.points[..., 2]

        # Create structured grid using the arbitrary lattice-based coordinates
        grid = pv.StructuredGrid(coord1, coord2, coord3)

        # Assign scalar field values to the grid
        grid["scalar_field"] = self.scalar_field.ravel(
            order='F')  # 'F' order to ensure correct reshaping

        # Display isosurface
        if isosurface_value is not None:
            contour = grid.contour(
                scalars="scalar_field", isosurfaces=[isosurface_value]
            )
            try:
                if smooth_surface:
                    contour = contour.subdivide(nsub=2, subfilter='loop')
                    contour = contour.smooth(n_iter=50)
                plotter.add_mesh(contour, color=isosurface_color,
                                 opacity=opacity, show_scalar_bar=False)
            except pv.core.errors.NotAllTrianglesError as e:
                # Calculate mean and standard deviation of scalar field
                mean_value = np.mean(self.scalar_field)
                std_dev = np.std(self.scalar_field)
                print(f"Error: Input mesh for subdivision must be all triangles.")
                print(
                    f"Your isovalue({isosurface_value}) may be far from the scalar field distribution.")
                print(f"Mean value of scalar field: {mean_value}")
                print(f"Standard deviation of scalar field: {std_dev}")
                mean_value = np.mean(np.abs(self.scalar_field))
                std_dev = np.std(np.abs(self.scalar_field))
                print(f"Mean value of absolute scalar field: {mean_value}")
                print(
                    f"Standard deviation of absolute scalar field: {std_dev}")

                raise e  # Re-raise the exception to fail the run

        # Display grid surface
        if show_grid_surface:
            plotter.add_mesh(grid.outline(), color=grid_surface_color)

        # Display grid points
        if show_grid_points:
            plotter.add_mesh(grid, style='points', point_size=grid_points_size,
                             color=grid_points_color, render_points_as_spheres=True)

        # Display filtered points based on value range
        if show_filtered_points:
            # Flatten the points array to correspond with the flattened scalar_field
            points_flattened = self.points.reshape(-1, 3)

            # Flatten the scalar_field array and create the boolean mask
            scalar_field = self.scalar_field.ravel()
            points_in_range = (scalar_field >= point_value_range[0]) & (
                scalar_field <= point_value_range[1])

            # Apply the boolean mask to select the points within the value range
            selected_points = points_flattened[points_in_range]

            # If there are points in the range, show them
            if len(selected_points) > 0:
                plotter.add_points(selected_points, color=grid_points_color,
                                   point_size=grid_points_size, render_points_as_spheres=True)
            else:
                print(f"No points found in the range {point_value_range}")

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
        info.append(f"lat1 (x): {self.lat3}")
        info.append(f"lat2 (y): {self.lat2}")
        info.append(f"lat3 (z): {self.lat1}")

        # Find the maximum value of scalar field and its coordinates
        max_density_idx = np.argmax(self.scalar_field)
        max_density_value = self.scalar_field.ravel()[max_density_idx]
        max_density_coords = np.unravel_index(
            max_density_idx, self.scalar_field.shape)
        max_density_point = self.points.reshape(
            *self.dimensions, 3)[max_density_coords]

        info.append(f"Maximum scalar field Value: {max_density_value}")
        info.append(
            f"Coordinates of Maximum scalar field: {max_density_point}")

        return '\n'.join(info)
