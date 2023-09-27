import numpy as np
import pyvista as pv
from .constants import BOHR_TO_ANGSTROM

DEFAULT_ELDENS_SETTINGS = {'show': True, 'isosurface_value': 0.1,
                           'show_grid_surface': False, 'show_grid_points': False}

class ElectronDensity:
    def __init__(self, electron_density, meta):
        self.electron_density = electron_density
        self.meta = meta

        # Derive points and dimensions from electron_density and meta
        nx, ny, nz = self.electron_density.shape
        self.dimensions = [nx, ny, nz]

        x = (np.linspace(0, nx - 1, nx) *
             self.meta['xvec'][0] + self.meta['org'][0])*BOHR_TO_ANGSTROM
        y = (np.linspace(0, ny - 1, ny) *
             self.meta['yvec'][1] + self.meta['org'][1])*BOHR_TO_ANGSTROM
        z = (np.linspace(0, nz - 1, nz) *
             self.meta['zvec'][2] + self.meta['org'][2])*BOHR_TO_ANGSTROM

        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        self.points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

        # # Print the parameters for debugging
        # print("Electron Density: ", self.electron_density)
        # print("Metadata: ", self.meta)
        # print("Points: ", self.points)
        # print("Dimensions: ", self.dimensions)

    @staticmethod
    def _getline(cube):
        l = cube.readline().strip().split()
        return int(l[0]), list(map(float, l[1:]))

    @staticmethod
    def read_cube(fname):
        meta = {}
        with open(fname, 'r') as cube:
            cube.readline()
            cube.readline()  # ignore comments
            natm, meta['org'] = ElectronDensity._getline(cube)
            nx, meta['xvec'] = ElectronDensity._getline(cube)
            ny, meta['yvec'] = ElectronDensity._getline(cube)
            nz, meta['zvec'] = ElectronDensity._getline(cube)
            meta['atoms'] = [ElectronDensity._getline(
                cube) for i in range(natm)]
            data = np.zeros((nx * ny * nz))
            idx = 0
            for line in cube:
                for val in line.strip().split():
                    data[idx] = float(val)
                    idx += 1
        data = np.swapaxes(np.reshape(data, (nx, ny, nz)), 0, -1)
        return data, meta

    @classmethod
    def load(cls, cube_file_path):
        # Read cube file and extract data and metadata
        data, meta = cls.read_cube(cube_file_path)

        # Return an instance of ElectronDensity initialized with data and meta
        return cls(data, meta)

    def rotate(self, rotation_matrix):
        """
        Rotates the points around the origin using the provided rotation matrix.

        :param rotation_matrix: 3x3 rotation matrix
        """
        assert rotation_matrix.shape == (
            3, 3), "Rotation matrix must be a 3x3 matrix."
        # Rotate each point
        self.points = np.dot(self.points, rotation_matrix.T)

    def translate(self, translation_vector):
        """
        Translates the points by the given translation vector.

        :param translation_vector: 1x3 translation vector
        """
        assert len(
            translation_vector) == 3, "Translation vector must be a 1x3 vector."
        self.points += translation_vector  # Translate each point

    def render(self, plotter=None, isosurface_value=None, isosurface_color='b', show_grid_surface=False, 
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

