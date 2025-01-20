from __future__ import annotations

import numpy as np
import pyvista as pv
from ase import Atoms

from .molecule import Molecule, fragment_vectors


def add_vector_to_plotter(p, r0, v, color='blue', scale_factor=1.0):
    # Create a single point from the origin
    points = pv.PolyData(r0)

    # Associate the orientation vector with the dataset
    # Making sure the vector is a 2D array with shape (n_points, 3)
    points['Vectors'] = np.array([v])

    # Generate arrows with scaling factor
    arrows = points.glyph(orient='Vectors', scale=True,
                          factor=scale_factor, geom=pv.Arrow())

    # Add arrows to the plotter
    p.add_mesh(arrows, color=color)


class Fragment(Molecule):
    def __init__(self, *args, attach_atom=None, attach_matrix=None, **kwargs):
        # If the first argument is an instance of Fragment
        if args and isinstance(args[0], Fragment):
            fragment = args[0]
            super().__init__(fragment)
            self.attach_atom = fragment.attach_atom
            self.attach_matrix = fragment.attach_matrix
        else:
            super().__init__(*args, **kwargs)
            if attach_atom is None:
                raise ValueError(
                    "attach_atom is mandatory unless copying from another Fragment")
            self.attach_atom = attach_atom
            self.attach_matrix = np.eye(
                3) if attach_matrix is None else attach_matrix

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as file:
            # Read number of atoms
            num_atoms = int(file.readline().strip())

            # Parse the "Fragment" line
            fragment_line = file.readline().strip().split()
            attach_atom = int(fragment_line[1])
            matrix_values = list(map(float, fragment_line[2:]))
            attach_matrix = np.array([matrix_values[i:i+3]
                                     for i in range(0, len(matrix_values), 3)])

            # Read Atoms
            symbols = []
            positions = []
            for _ in range(num_atoms):
                line = file.readline().strip().split()
                atom, x, y, z = line[0], float(
                    line[1]), float(line[2]), float(line[3])
                symbols.append(atom)
                positions.append([x, y, z])

            # Initialize the Atoms object using ase
            atoms = Atoms(symbols=symbols, positions=positions)

            return cls(atoms, attach_atom=attach_atom, attach_matrix=attach_matrix)

    @property
    def attach_point(self):
        return self.get_positions()[self.attach_atom]

    @property
    def fragment_vectors(self):
        return fragment_vectors(self.attach_atom, self.get_positions())

    def copy(self):
        return Fragment(self, self.attach_atom, self.attach_matrix)

    def reorder(self, mapping: dict):
        """
        Reorders the atoms in the Fragment instance based on the provided mapping and updates the attach_atom.

        Parameters:
            - mapping (dict): A dictionary where keys are current indices and values are target indices.

        Returns:
            - Fragment: A new Fragment instance with atoms reordered based on the mapping.
        """
        reordered_molecule = super().reorder(
            mapping)  # Call the reorder method from the Molecule class

        # Update the attach_atom if it's present in the mapping
        if self.attach_atom in mapping:
            new_attach_atom = mapping[self.attach_atom]
        else:
            new_attach_atom = self.attach_atom

        # Convert the reordered molecule to a Fragment instance and return
        return Fragment(reordered_molecule, attach_atom=new_attach_atom, attach_matrix=self.attach_matrix)

    def main_vectors_render(self, plotter: pv.Plotter = None, notebook=False):
        if plotter is None:
            plotter = pv.Plotter(notebook=notebook)
        V0, V1, V2, V3 = self.fragment_vectors
        add_vector_to_plotter(plotter, V0, V1, color='red', scale_factor=1.0)
        add_vector_to_plotter(plotter, V0, V2, color='green', scale_factor=0.7)
        add_vector_to_plotter(plotter, V0, V3, color='blue', scale_factor=0.7)
        return plotter

    def get_main_vectors(self):
        V0 = self.attach_point
        V1 = self.attach_matrix[:, 0]
        V2 = self.attach_matrix[:, 1]
        V3 = self.attach_matrix[:, 2]
        return V0, V1, V2, V3

    def main_vectors_render(self, plotter: pv.Plotter = None, notebook=False):
        if plotter is None:
            plotter = pv.Plotter(notebook=notebook)
        V0, V1, V2, V3 = self.get_main_vectors()
        add_vector_to_plotter(plotter, V0, V1, color='red', scale_factor=1.0)
        add_vector_to_plotter(plotter, V0, V2, color='green', scale_factor=0.7)
        add_vector_to_plotter(plotter, V0, V3, color='blue', scale_factor=0.7)
        return plotter

    def apply_transition(self, r0):
        self.set_positions(self.get_positions() + r0)

    def apply_rotation(self, R):
        # Rotate atom positions (assuming each row is a position vector)
        self.set_positions(np.dot(R, self.get_positions().T).T)

        # Rotate the attach_matrix (assuming each column is a vector)
        self.attach_matrix = np.dot(R, self.attach_matrix)

    def get_origin_rotation_matrix(self):
        return np.linalg.inv(self.attach_matrix)

    def set_to_origin(self):
        self.apply_transition(-self.attach_point)
        self.apply_rotation(self.get_origin_rotation_matrix())

    def render(self, **kwargs):
        if 'plotter' in kwargs:
            plotter = self.main_vectors_render(kwargs['plotter'])
        else:
            notebook = kwargs.get('Notebook', False)
            plotter = self.main_vectors_render(None, notebook=notebook)

        # Update the plotter in kwargs before calling super
        kwargs['plotter'] = plotter
        return super().render(**kwargs)

    def render_alongside(self, other, other_alpha=1.0, **kwargs):
        # Get the plotter from kwargs or default to None
        plotter = kwargs.get('plotter', None)

        # Process the plotter using main_vectors_render
        plotter = self.main_vectors_render(plotter)

        # Update the plotter in kwargs
        kwargs['plotter'] = plotter

        # Call the super method
        return super().render_alongside(other, other_alpha=other_alpha, **kwargs)

    def to_xyz_string(self):
        # Convert attach matrix to string
        matrix_string = ' '.join([' '.join(map(str, row))
                                  for row in self.attach_matrix])

        # Create the second line with embedded info
        fragment_info = f'Fragment: {self.attach_atom} {matrix_string}'

        # Construct the save string
        atom_info = super().to_xyz_string().split('\n')
        main_part = '\n'.join(atom_info[2:])
        return f"{atom_info[0]}\n{fragment_info}\n{main_part}"

    def connect_fragment(self, fragment: Fragment, check_bonds_quantity=False):
        _fragment = fragment.copy()
        _fragment.set_to_origin()
        _fragment.apply_rotation(self.attach_matrix)
        _fragment.apply_transition(self.attach_point)
        molecule = self.copy()
        del molecule[molecule.attach_atom]
        molecule = Molecule(molecule)
        molecule.extend(_fragment)
        if check_bonds_quantity:
            scale_factor = 0.01
            while len(molecule.get_all_bonds()) != len(self.get_all_bonds()) + len(_fragment.get_all_bonds()):
                print(
                    f'{len(molecule.get_all_bonds())=} {len(self.get_all_bonds()) + len(_fragment.get_all_bonds())=}')
                scale_factor += 0.01
                print(f'{scale_factor=}')
                _fragment = fragment.copy()
                _fragment.set_to_origin()
                random_matrix = np.random.rand(3, 3) * scale_factor
                attach_matrix = self.attach_matrix+random_matrix
                _fragment.apply_rotation(attach_matrix)
                _fragment.apply_transition(self.attach_point)
                molecule = self.copy()
                del molecule[molecule.attach_atom]
                molecule = Molecule(molecule)
                molecule.extend(_fragment)

        return molecule
