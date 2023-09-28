import numpy as np
import pyvista as pv
from .constants import BOHR_TO_ANGSTROM
from scipy.interpolate import griddata


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
        l = cube.readline().strip().split()
        return int(l[0]), list(map(float, l[1:]))

    @staticmethod
    def read_cube(fname):
        """
        Reads a cube file and extracts metadata and volumetric data.

        Parameters:
            fname (str): Path to the cube file.

        Returns:
            numpy.ndarray: A 3D array containing the volumetric data.
            dict: A dictionary containing metadata extracted from the cube file.
        """
        meta = {}
        with open(fname, 'r') as cube:
            cube.readline()  # Skip the first two comment lines in the cube file
            cube.readline()

            # Read metadata: number of atoms (natm) and origin (meta['org'])
            natm, meta['org'] = ElectronDensity._getline(cube)

            # Read the number of points and vector information in each dimension
            nx, meta['xvec'] = ElectronDensity._getline(cube)
            ny, meta['yvec'] = ElectronDensity._getline(cube)
            nz, meta['zvec'] = ElectronDensity._getline(cube)

            # Convert from Bohr to Angstroms for origin and vectors
            meta['org'] = [x * BOHR_TO_ANGSTROM for x in meta['org']]
            meta['xvec'] = [x * BOHR_TO_ANGSTROM for x in meta['xvec']]
            meta['yvec'] = [y * BOHR_TO_ANGSTROM for y in meta['yvec']]
            meta['zvec'] = [z * BOHR_TO_ANGSTROM for z in meta['zvec']]

            # Extract atom information, considering the absolute value of natm, as natm can be negative for molecular orbitals
            meta['atoms'] = [ElectronDensity._getline(
                cube) for _ in range(abs(natm))]

            data_values = []
            firstline = True

            for line in cube:
                values_in_line = [float(val) for val in line.strip().split()]
                if firstline:
                    firstline = False
                    # If the first line contains two elements, it is considered as orbital info and skipped
                    if len(values_in_line) == 2:
                        continue
                # Extend the list with the actual data values
                data_values.extend(values_in_line)

            # Check if the read data points match the expected data points
            if len(data_values) != nx * ny * nz:
                raise ValueError(
                    f"Number of data points in the file ({len(data_values)}) does not match the expected size ({nx * ny * nz})")

            # Reshape the 1D list of data_values to a 3D array and swap axes to match the expected orientation
            data = np.array(data_values).reshape((nx, ny, nz))
            data = np.swapaxes(data, 0, -1)

        return data, meta

    @classmethod
    def load(cls, cube_file_path):
        # Read cube file and extract data and essential metadata
        data, meta = cls.read_cube(cube_file_path)

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
               grid_points_color="r", grid_points_size=5, save=None, show=False):

        if plotter is None:
            plotter = pv.Plotter(notebook=notebook)

        nx, ny, nz = self.dimensions
        x, y, z = self.points.T.reshape(3, nx, ny, nz)
        grid = pv.StructuredGrid(x, y, z)

        if isosurface_value is not None:
            plotter.add_mesh(grid.contour(scalars=self.electron_density.ravel(),
                                          isosurfaces=[isosurface_value]),
                             color=isosurface_color, opacity=opacity, show_scalar_bar=False)

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
