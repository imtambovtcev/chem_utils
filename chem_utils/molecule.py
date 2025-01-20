from __future__ import annotations

import io
import math
import pathlib
import warnings
from collections import Counter
from io import BytesIO
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import pyperclip
import pyvista as pv
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.io import read, write
from IPython.display import Image, display
from rdkit import Chem
from rdkit.Chem import AddHs, CanonicalRankAtoms, Draw, GetSSSR, SanitizeMol
from rdkit.Chem.rdDepictor import Compute2DCoords
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from .constants import BOHR_TO_ANGSTROM

from .constants import BOHR_TO_ANGSTROM
from .scalar_field import ScalarField
from .valency import add_valency, general_print, rebond

DEFAULT_ATOMS_SETTINGS = pd.read_csv(
    str(pathlib.Path(__file__).parent/'pyvista_render_settings.csv'))
DEFAULT_ATOMS_SETTINGS.index = list(DEFAULT_ATOMS_SETTINGS['Name'])
DEFAULT_ATOMS_SETTINGS['Color'] = [[int(i) for i in s.replace(
    '[', '').replace(']', '').split(',')] for s in DEFAULT_ATOMS_SETTINGS['Color']]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    return tuple(int(hex_color[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def rotation_matrix_from_points(m0, m1):
    """Returns a rigid transformation/rotation matrix that minimizes the
    RMSD between two set of points.

    m0 and m1 should be (3, npoints) numpy arrays with
    coordinates as columns::

        (x1  x2   x3   ... xN
         y1  y2   y3   ... yN
         z1  z2   z3   ... zN)

    The centeroids should be set to origin prior to
    computing the rotation matrix.

    The rotation matrix is computed using quaternion
    algebra as detailed in::

        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """

    v0 = np.copy(m0)
    v1 = np.copy(m1)

    # compute the rotation quaternion

    R11, R22, R33 = np.sum(v0 * v1, axis=1)
    R12, R23, R31 = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
    R13, R21, R32 = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)

    f = [[R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21],
         [R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31],
         [R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32],
         [R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33]]

    F = np.array(f)

    w, V = np.linalg.eigh(F)
    # eigenvector corresponding to the most
    # positive eigenvalue
    q = V[:, np.argmax(w)]

    # Rotation matrix from the quaternion q

    R = quaternion_to_matrix(q)

    return R


def quaternion_to_matrix(q):
    """Returns a rotation matrix.

    Computed from a unit quaternion Input as (4,) numpy array.
    """

    q0, q1, q2, q3 = q
    R_q = [[q0**2 + q1**2 - q2**2 - q3**2,
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2)],
           [2 * (q1 * q2 + q0 * q3),
            q0**2 - q1**2 + q2**2 - q3**2,
            2 * (q2 * q3 - q0 * q1)],
           [2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            q0**2 - q1**2 - q2**2 + q3**2]]
    return np.array(R_q)


def fragment_vectors(i, coords):
    V0 = coords[i]
    n_unique_nodes = len(coords)

    # Single Point Case
    if n_unique_nodes == 1:
        warnings.warn(
            "Only one point provided. Using arbitrary perpendicular vectors.")
        V1 = np.array([1, 0, 0])
        V2 = np.array([0, 1, 0])
        V3 = np.array([0, 0, 1])
        return V0, V1, V2, V3

    # Two Points Case
    if n_unique_nodes == 2:
        warnings.warn(
            "Only two points provided. Adjusting vectors based on reference vectors.")
        V1 = coords[(i-1) % 2] - coords[i]
        V1 = V1 / np.linalg.norm(V1)

        # If V1 is essentially the x-axis, set V2 to y-axis
        if np.allclose(V1, [1, 0, 0], atol=1e-8):
            V2 = np.array([0, 1, 0])
            V3 = np.array([0, 0, 1])
        else:
            reference_vectors = [np.array([1, 0, 0]), np.array(
                [0, 1, 0]), np.array([0, 0, 1])]
            dot_products = [np.abs(np.dot(V1, ref_vec))
                            for ref_vec in reference_vectors]

            # Exclude the vector most aligned with V1
            exclude_index = np.argmax(dot_products)
            remaining_vectors = [v for i, v in enumerate(
                reference_vectors) if i != exclude_index]

            # Check cross products with the remaining vectors and choose V2 such that V3 aligns most with the excluded vector
            cross1 = np.cross(V1, remaining_vectors[0])
            cross2 = np.cross(V1, remaining_vectors[1])

            if np.dot(cross1, reference_vectors[exclude_index]) > np.dot(cross2, reference_vectors[exclude_index]):
                V2 = remaining_vectors[0]
                V3 = cross1
            else:
                V2 = remaining_vectors[1]
                V3 = cross2

        return V0, V1, V2, V3

    # Three or More Points Case
    pca = PCA(n_components=3)
    pca.fit(coords)

    explained_variances = pca.explained_variance_
    components = pca.components_

    ordered_indices = np.argsort(explained_variances)[::-1]
    V1 = components[ordered_indices[0]]
    V2 = components[ordered_indices[1]]
    V3 = np.cross(V1, V2)

    if n_unique_nodes == 3 and np.linalg.matrix_rank(coords) == 2:
        warnings.warn(
            "The three points provided are collinear. Adjusting vectors based on PCA.")

    return V0, V1, V2, V3


def adjust_bond_length(atom_radius, offset_distance, cylinder_radius):
    # Calculate additional length for atom
    x = abs(offset_distance) + cylinder_radius
    L = atom_radius-np.sqrt(atom_radius**2 - x**2)
    return L


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class Molecule(Atoms):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar_fields = {}  # Placeholder for the ScalarField objects
        self.reinit()

    def reinit(self):
        self.ana = Analysis(self)
        self.G = nx.Graph()  # Start with an empty graph

        for idx, symbol in enumerate(self.get_chemical_symbols()):
            self.G.add_node(idx, label=symbol)

        # Get the bonds from self.ana.all_bonds
        all_bonds = self.ana.all_bonds[0]
        unique_bonds = []
        for i, neighbors in enumerate(all_bonds):
            for j in neighbors:
                # To ensure uniqueness, we'll only consider bonds where i < j
                if i < j:
                    unique_bonds.append((i, j))

        self.G.add_edges_from(unique_bonds)

    def __delitem__(self, i):
        # Call the original __delitem__ method from the Atoms class to handle removal
        super().__delitem__(i)
        # Reinitialize to reflect the changes
        self.reinit()

    def bond_print(self):
        return general_print(self.G)

    def add_valency(self):
        self.G = add_valency(self.G)

    def bond_assertion(self):
        for node, data in self.G.nodes(data=True):
            atom_symbol = data['label']
            atom_valency = data['valency']

            # Count the number of bonds/edges connected to the node
            bond_count = self.G.degree(node)
            assert bond_count <= atom_valency, f"""Atom {node} ({atom_symbol})
                has {bond_count} bonds, which exceeds its known valency of {atom_valency}!"""

    def update_bond_labels(self):
        self.add_valency()
        self.bond_assertion()
        H = rebond(self.G)
        for i in H.nodes():
            if H.nodes[i]['valency'] == 0:
                for j in H.neighbors(i):
                    self.G.edges[i, j]['bond_type'] = H.edges[i,
                                                              j]['bond_type']

    @classmethod
    def load(cls, filename):
        # Check if the file format is 'cube'
        if str(filename).lower().endswith('.cube'):
            # Display a warning if the format is 'cube'
            warnings.warn(
                "To get the surface information, use the 'load_from_cube' method.")

        # Use ASE's read function to get Atoms object from file
        atoms = read(filename)

        # Create a Molecule instance from the Atoms object
        molecule = cls(atoms)

        return molecule

    @classmethod
    def load_from_cube(cls, cube_file_path, name='Electron density'):
        # Use the ElectronDensity class method to read cube file
        data, meta = ScalarField.read_cube(
            cube_file_path)

        # Extract atoms information from meta
        atoms_info = meta['atoms']
        numbers = [atomic_number for atomic_number, _ in atoms_info]
        positions = [coordinates for _, coordinates in atoms_info]

        # Create a Molecule instance with the extracted atomic numbers and positions
        molecule = cls(numbers=numbers, positions=positions)

        # Create an ScalarField instance and adds it to the molecule's scalar_fields dictionary
        molecule.scalar_fields[name] = ScalarField(
            data, meta['org'], meta['lat1'], meta['lat2'], meta['lat3'])

        return molecule

    @classmethod
    def load_from_gpaw(cls, filename: str | pathlib.Path, coverage="all"):
        from gpaw import restart

        def get_orbital_range(calc, coverage="all"):
            """
            Determine the range of orbitals for a given GPAW calculation.

            Parameters:
            calc (GPAW calculator object): GPAW calculation with wavefunctions.
            coverage (str): Specify which orbitals to cover. Options are:
                            "all" - Covers all orbitals (default).
                            "occupied" - Covers only the occupied orbitals.
                            "unoccupied" - Covers only the unoccupied orbitals.

            Returns:
            tuple: (i, a) where 'i' is the offset from HOMO, and 'a' is the number of orbitals above LUMO.
            """
            f_n = calc.wfs.kpt_u[0].f_n  # Assuming spin-unpolarized case for simplicity
            nlumo = len(f_n[f_n > 0])
            nhomo = nlumo - 1

            if coverage == "all":
                i = -nhomo  # Start from the lowest occupied orbital
                a = len(f_n) - nlumo-1  # Include all unoccupied orbitals
            elif coverage == "occupied":
                i = -nhomo  # Start from the lowest occupied orbital
                a = 0  # Do not include any unoccupied orbitals
            elif coverage == "unoccupied":
                i = 0  # Start from the LUMO
                a = len(f_n) - nlumo-1  # Include all unoccupied orbitals
            else:
                raise ValueError(
                    "Invalid coverage option. Choose 'all', 'occupied', or 'unoccupied'.")

            return i, a
        # Restart GPAW calculation
        atoms, calc = restart(str(filename), txt=None)
        molecule = cls(atoms)

        i, a = get_orbital_range(calc, coverage=coverage)

        all_electron_density = calc.get_all_electron_density(gridrefinement=1)
        org = np.array([0., 0., 0.])
        lat1 = calc.wfs.gd.h_cv[0]*BOHR_TO_ANGSTROM
        lat2 = calc.wfs.gd.h_cv[1]*BOHR_TO_ANGSTROM
        lat3 = calc.wfs.gd.h_cv[2]*BOHR_TO_ANGSTROM

        molecule.scalar_fields['All electron density'] = ScalarField(
            all_electron_density, org, lat1, lat2, lat3)

        for s in range(1):
            f_n = calc.wfs.kpt_u[s].f_n

            # Determine indices of HOMO and LUMO
            nlumo = len(f_n[f_n > 0])
            nhomo = nlumo - 1
            for n in range(nhomo + i, nlumo + a + 1):
                # Retrieve the wavefunction for the given orbital
                orb = calc.get_pseudo_wave_function(band=n, spin=s)
                molecule.scalar_fields[f'Orbital {n}'] = ScalarField(
                    orb, org, lat1, lat2, lat3)

        return molecule

    def __eq__(self, other: Atoms):
        """
        Molecules are equal if they have the same atoms
        """
        # Check if both are instances of Atoms
        if not isinstance(other, Atoms):
            return False

        # Get the counts of each atom type
        self_counts = Counter(self.get_chemical_symbols())
        other_counts = Counter(other.get_chemical_symbols())

        # Check if the atom type counts are the same
        return self_counts == other_counts

    def copy(self):
        """Return a copy of the Molecule."""
        new_molecule = self.__class__()
        for key, value in self.arrays.items():
            new_molecule.arrays[key] = value.copy()
        new_molecule.set_cell(self.get_cell().copy())
        new_molecule.set_pbc(self.get_pbc().copy())
        new_molecule.info = self.info.copy()
        for key, value in self.scalar_fields.items():
            new_molecule.scalar_fields[key] = value.copy()
        return new_molecule

    def extend(self, atoms):
        super().extend(atoms)
        self.reinit()

    def set_positions_no_reinit(self, new_coords, atoms=None):
        if atoms is None:
            super().set_positions(new_coords)
        else:
            co = self.get_positions()
            co[atoms, :] = new_coords
            super().set_positions(co)
        self.reinit()

    def set_positions(self, new_coords, atoms=None):
        self.set_positions_no_reinit(new_coords, atoms)
        self.reinit()

    def translate(self, translation_vector):
        """
        Translates the points by the given translation vector.

        :param translation_vector: 1x3 translation vector
        """
        assert len(
            translation_vector) == 3, "Translation vector must be a 1x3 vector."
        self.positions += translation_vector  # Translate each point
        for field in self.scalar_fields.values():
            field.translate(translation_vector)

    def rotate(self, rotation_matrix):
        """
        Rotates the points around the origin using the provided rotation matrix.

        :param rotation_matrix: 3x3 rotation matrix
        """
        assert rotation_matrix.shape == (
            3, 3), "Rotation matrix must be a 3x3 matrix."
        # Rotate each point
        self.positions = np.dot(self.positions, rotation_matrix.T)
        for field in self.scalar_fields.values():
            field.rotate(rotation_matrix)

    def to_origin(self, zero_atom=None, x_atom=None, no_z_atom=None):
        """
        Move molecule by 3 atoms:
        - zero_atom: to be placed to (0,0,0)
        - x_atom: to be placed to (?, 0, 0)
        - no_z_atom: to be placed to (?, ?, 0)
        """
        if all(variable is None for variable in (zero_atom, x_atom, no_z_atom)):
            zero_atom = 0

        self.translate(-self.positions[zero_atom, :])
        if x_atom is not None:
            rm = rotation_matrix(np.cross(self.positions[x_atom, :], [1, 0, 0]), np.arccos(
                np.dot(self.positions[x_atom, :], [1, 0, 0])/np.linalg.norm(self.positions[x_atom, :])))
            self.rotate(rm)
        if no_z_atom is not None:
            rm = rotation_matrix(np.array(
                [1., 0., 0.]), -np.arctan2(self.positions[no_z_atom, 2], self.positions[no_z_atom, 1]))
            self.rotate(rm)

    def rotate_part(self, atom, bond, angle):
        edge = bond if bond[0] == atom else bond[::-1]
        r0 = self.get_positions()[atom]
        v = self.get_positions()[edge[1]]-r0
        island = self.divide_in_two(edge)[0]
        r = rotation_matrix(v, np.pi*angle/180)
        new_coords = np.matmul(r, (self.get_positions()[island]-r0).T).T+r0
        self.set_positions(new_coords, island)
        return self

    def shake(self, amplitude=0.05):
        """
        Randomly displace the positions of all atoms in the molecule.

        Parameters:
        - amplitude (float): The maximum magnitude of displacement in any direction.
        """
        # Get the current positions of the atoms
        positions = self.get_positions()

        # Generate random displacements for each atom
        displacements = np.random.uniform(-amplitude,
                                          amplitude, positions.shape)

        # Apply the displacements to the atom positions
        new_positions = positions + displacements

        # Set the new positions
        self.set_positions(new_positions)

    def displace_along_mode(self, mode: np.ndarray, amplitude: float):
        """
        Displace each atom of the molecule along the given mode (vibration vector) 
        by a specified amplitude (magnitude).

        Args:
            mode (np.ndarray): A numpy array with shape (n_atoms, 3) representing 
                the direction of displacement for each atom.
            amplitude (float): The magnitude of the displacement along the mode.
        """
        # Validate the mode argument
        if not isinstance(mode, np.ndarray) or mode.shape != (len(self), 3):
            raise ValueError(
                "mode must be a numpy array with shape (n_atoms, 3)")

        if not isinstance(amplitude, (int, float)):
            raise ValueError("amplitude must be a numeric value")

        # Calculate the new positions of the atoms after displacement along the mode
        new_positions = self.get_positions() + amplitude * mode

        # Set the new positions using the set_positions method of the Molecule class
        self.set_positions(new_positions)

    def divide_in_two(self, bond):
        G = self.G.copy()
        G.remove_edge(bond[0], bond[1])
        return list(nx.shortest_path(G, bond[0]).keys()), list(nx.shortest_path(G, bond[1]).keys())

    def divide_in_two_fragments(self, fragment_attach_atom, fragment_atoms_ids, vector_mode='both'):
        """
        fragment_attach_atom - atom in fragment that is attached to the rest of the molecule
        fragment_atoms_ids - ids of the atoms fragments
        """
        from .fragment import Fragment
        small_fragment_ids = list(set(fragment_atoms_ids))
        main_fragment_ids = list(
            (set(range(len(self)))-set(small_fragment_ids)) | {fragment_attach_atom})
        small_fragment_bonds = [bond for bond in self.get_bonds_of(
            small_fragment_ids) if set(bond).issubset(set(small_fragment_ids))]
        main_fragment_bonds = [
            bond for bond in self.get_all_bonds() if bond not in small_fragment_bonds]
        new_small_fragment_ids = dict(
            zip(small_fragment_ids, range(len(small_fragment_ids))))
        new_main_fragment_ids = dict(
            zip(main_fragment_ids, range(len(main_fragment_ids))))
        _V0_s, _V1_s, _V2_s, _V3_s = fragment_vectors(
            new_small_fragment_ids[fragment_attach_atom], self.get_positions()[small_fragment_ids, :])
        _V0_m, _V1_m, _V2_m, _V3_m = fragment_vectors(
            new_main_fragment_ids[fragment_attach_atom], self.get_positions()[main_fragment_ids, :])

        if vector_mode == 'both':
            V1_s, V2_s, V3_s = _V1_s, _V2_s, _V3_s
            V1_m, V2_m, V3_m = _V1_m, _V2_m, _V3_m
        elif vector_mode == 'main':
            V1_s, V2_s, V3_s = -_V1_m, -_V2_m, -_V3_m
            V1_m, V2_m, V3_m = _V1_m, _V2_m, _V3_m
        elif vector_mode == 'small':
            V1_s, V2_s, V3_s = _V1_s, _V2_s, _V3_s
            V1_m, V2_m, V3_m = -_V1_s, -_V2_s, -_V3_s

        R_s = np.column_stack([V1_s, V2_s, V3_s])
        R_m = np.column_stack([V1_m, V2_m, V3_m])
        # R_s = np.linalg.inv(np.column_stack([V1_s, V2_s, V3_s]))
        # R_m = np.linalg.inv(np.column_stack([V1_m, V2_m, V3_m]))
        # R_to_small = np.dot(R_s, np.linalg.inv(R_m))
        # R_to_main = np.linalg.inv(R_to_small)
        small_atoms = Atoms(
            self.symbols[small_fragment_ids], positions=self.get_positions()[small_fragment_ids, :])
        main_atoms = Atoms(
            self.symbols[main_fragment_ids], positions=self.get_positions()[main_fragment_ids, :])
        return Fragment(main_atoms, attach_atom=new_main_fragment_ids[fragment_attach_atom], attach_matrix=R_s), Fragment(small_atoms, attach_atom=new_small_fragment_ids[fragment_attach_atom], attach_matrix=R_m)

    def compare(self, other):
        return self.symbols

    def get_all_bonds(self):
        return list(self.G.edges())

    def get_bonded_atoms_of(self, i):
        return list(self.G[i])

    def get_bonds_of(self, i):
        if isinstance(i, int):
            return [bond for bond in self.get_all_bonds() if i in self.get_all_bonds()]
        else:
            return [bond for bond in self.get_all_bonds() if len(set(bond) & set(i)) > 0]

    def id_to_symbol(self, l):
        try:
            return self.G.nodes[l]['label']
        except:
            return [self.G.nodes[i]['label'] for i in l]

    def spatial_distance_between_atoms(self, i, j):
        if i is None or j is None:
            warnings.warn("Either 'i' or 'j' is None. Returning NaN.")
            return np.NaN
        c = self.get_positions()
        return np.linalg.norm(c[i, :]-c[j, :])

    def to_rdkit_mol(self):
        """Convert this Molecule instance to an RDKit Mol object."""
        # Create an empty editable molecule
        mol = Chem.EditableMol(Chem.Mol())

        # Add atoms to the molecule
        for symbol in self.get_chemical_symbols():
            atom = Chem.Atom(symbol)
            mol.AddAtom(atom)

        # Add bonds to the molecule based on self.G
        for (i, j) in self.get_all_bonds():
            label = self.G.edges[i, j].get('bond_type', 0)
            if label == 1:
                bond_type = Chem.BondType.SINGLE
            elif label == 2:
                bond_type = Chem.BondType.DOUBLE
            elif label == 3:
                bond_type = Chem.BondType.TRIPLE
            else:
                bond_type = Chem.BondType.SINGLE  # default to single if 0 or not specified
            mol.AddBond(int(i), int(j), bond_type)

        # Convert editable molecule to a regular Mol object and return
        return mol.GetMol()

    @staticmethod
    def from_rdkit_mol(mol):
        """Create a Molecule instance from an RDKit Mol object."""
        atoms = Atoms(len(mol.GetAtoms()),
                      positions=[atom.GetPos() for atom in mol.GetAtoms()],
                      symbols=[atom.GetSymbol() for atom in mol.GetAtoms()])
        return Molecule(atoms)

    def is_isomorphic(self, other, respect_labels=True):
        if respect_labels:
            # Define a function to check if node labels are equal
            def node_match(n1, n2):
                return n1['label'] == n2['label']
            return nx.is_isomorphic(self.G, other.G, node_match=node_match)
        else:
            return nx.is_isomorphic(self.G, other.G)

    def all_mappings(self, other, respect_labels=True):
        if respect_labels:
            # Use the node_label argument directly for vf2pp_all_isomorphisms
            if not nx.is_isomorphic(self.G, other.G, node_match=lambda n1, n2: n1['label'] == n2['label']):
                warnings.warn(
                    "The structures are not isomorphic. Returning None.", UserWarning)
                return None
            all_iso = nx.vf2pp_all_isomorphisms(
                self.G, other.G, node_label='label')
        else:
            if not nx.is_isomorphic(self.G, other.G):
                return None
            all_iso = nx.vf2pp_all_isomorphisms(self.G, other.G)

        return all_iso

    def minimize_rmsd(self, other: Molecule):
        atoms = other.copy()
        p = atoms.get_positions()
        p0 = self.get_positions()

        # centeroids to origin
        c = np.mean(p, axis=0)
        p -= c
        c0 = np.mean(p0, axis=0)
        p0 -= c0

        # Compute rotation matrix
        R = rotation_matrix_from_points(p.T, p0.T)

        atoms.translate(-c)
        atoms.rotate(R.T)
        atoms.translate(c0)

        return atoms

    def iso_distance(self, other, iso=None, method='cdist'):
        coord_1 = self.get_positions()
        coord_2 = other.get_positions()

        # Check for NaN in initial coordinates
        if np.any(np.isnan(coord_1)) or np.any(np.isnan(coord_2)):
            warnings.warn("NaN found in initial coordinates")
            return np.nan

        if iso is not None:
            coord_1 = coord_1[np.array(list(iso.keys()), dtype=int), :]
            coord_2 = coord_2[np.array(list(iso.values()), dtype=int), :]

        if method == 'cdist':
            distance_matrix_diff = np.linalg.norm(
                cdist(coord_1, coord_1) - cdist(coord_2, coord_2))
            if np.isnan(distance_matrix_diff):
                warnings.warn("NaN found in distance matrix difference")
            return distance_matrix_diff

        # Centeroids to origin
        c = np.mean(coord_2, axis=0)
        coord_2 -= c
        c0 = np.mean(coord_1, axis=0)
        coord_1 -= c0

        # Compute rotation matrix
        R = rotation_matrix_from_points(coord_2.T, coord_1.T)

        if R is None or np.any(np.isnan(R)):
            warnings.warn("Rotation matrix computation failed or contains NaN")
            return np.nan

        coord_2 = (np.dot(coord_2, R.T) + c0)
        coord_1 += c0

        dist2 = np.sum((coord_1 - coord_2) ** 2, axis=-1)
        final_distance = np.sqrt(((coord_1 - coord_2) ** 2).mean())

        if np.isnan(final_distance):
            print("NaN found in final distance calculation")
            print("coord_1:", coord_1)
            print("coord_2:", coord_2)
            print("Final distance:", final_distance)

        return final_distance

    def best_mapping(self, other, limit=None, method='cdist'):
        if not (self.is_isomorphic(other)):
            warnings.warn(
                "The structures are not isomorphic. Returning None.", UserWarning)
            return None
        isos = list(islice(self.all_mappings(other), limit))
        return min(isos, key=lambda x: self.iso_distance(other, x, method=method))

    def best_distance(self, other: Atoms, limit=None, method='cdist'):
        if self.is_isomorphic(other):
            return self.iso_distance(other, iso=self.best_mapping(
                other, limit=limit, method=method), method=method)
        else:
            warnings.warn(
                "The structures are not isomorphic. Returning None.", UserWarning)
            return np.nan

    def reorder(self, mapping: dict):
        """
        Reorders the atoms in the molecule based on the provided mapping.

        Parameters:
            - mapping (dict): A dictionary where keys are current indices and values are target indices.

        Returns:
            - Molecule: A new molecule with atoms reordered based on the mapping.
        """
        # Initialize an order array with current ordering
        order = np.arange(len(self))

        # Update the order array based on the mapping
        for current_index, target_index in mapping.items():
            order[current_index] = target_index

        # Reorder the atomic positions and symbols
        reordered_coords = self.get_positions()[order, :]
        reordered_symbols = np.array(self.get_chemical_symbols())[
            order].tolist()

        # Return the new reordered molecule
        return Molecule(symbols=reordered_symbols, positions=reordered_coords)

    def get_standard_order(self):
        """Returns a reordered copy of the molecule based on RDKit's standard order."""
        mol = self.to_rdkit_mol()

        # Sanitize the molecule to calculate implicit valence and other properties
        SanitizeMol(mol)

        # Note down the number of atoms in the original molecule
        num_atoms_original = mol.GetNumAtoms()

        # Add implicit hydrogens
        mol = AddHs(mol)

        # Calculate SSSR
        GetSSSR(mol)

        # Get the canonical atom order from RDKit
        canonical_order = CanonicalRankAtoms(mol)

        # Filter canonical_order to match the size of the original molecule
        canonical_order = [
            index for index in canonical_order if index < num_atoms_original]

        mapping = dict(enumerate(canonical_order))

        # Use the reorder method to reorder atoms based on the canonical order
        reordered_molecule = self.reorder(mapping)
        return reordered_molecule

    def get_bonds_without(self, elements=['H', 'F']):
        # Get nodes that correspond to the undesired elements
        undesired_nodes = [node for node, attrs in self.G.nodes(
            data=True) if attrs['label'] in elements]

        # Create a subgraph by removing undesired nodes
        subgraph = self.G.copy()
        subgraph.remove_nodes_from(undesired_nodes)

        # Return all edges of the subgraph
        return list(subgraph.edges())

    def render(self, plotter: pv.Plotter = None, show=False, save=None, save_3d=None, atoms_settings=None, show_hydrogens=True, alpha=1.0, atom_numbers=False, show_hydrogen_bonds=False, show_numbers=False, show_basis_vectors=False, cpos=None, notebook=False, auto_close=True, interactive=True, background_color='black', valency=False, resolution=20,  light_settings=None, mode=None, scalar_fields_settings=None):
        """
        Renders a 3D visualization of a molecule using the given settings.

        Args:
            plotter (pv.Plotter, optional): The PyVista plotter object used for rendering. If not provided, a new one will be created. Defaults to None.
            atoms_settings (DataFrame, optional): A dataframe containing the visualization settings for each atom type. Defaults to DEFAULT_ATOMS_SETTINGS.
            show_hydrogens (bool, optional): Whether to render hydrogen atoms. Defaults to True.
            alpha (float, optional): The opacity of the rendered atoms and bonds. Value should be between 0 (transparent) and 1 (opaque). Defaults to 1.0.
            atom_numbers (bool, optional): Whether to display atom numbers. Defaults to False.
            show_hydrogen_bonds (bool, optional): Whether to visualize hydrogen bonds. Defaults to False.
            show_numbers (bool, optional): Whether to show atom indices. Defaults to False.
            show_basis_vectors (bool, optional): Whether to display the basis vectors. Defaults to True.
            save (str or bool, optional): Filepath to save the rendered image. If provided, the rendered image will be saved to this path, if bool True and no plotter provided, will create a plotter for screenshot. Defaults to None.
            save_3d (str, optional): Filepath to save the entire scene as a 3D model. The file format is determined by the file extension. Supported formats include '.ply', '.stl', etc. Defaults to None.
            cpos (list or tuple, optional): Camera position for the PyVista plotter. Defaults to None.
            notebook (bool, optional): Whether the function is being called within a Jupyter notebook. Adjusts the rendering accordingly. Defaults to False.
            auto_close (bool, optional): Whether to automatically close the rendering window after saving or completing the rendering. Defaults to True.
            interactive (bool, optional): Whether to allow interactive visualization (e.g., rotation, zoom). Defaults to False.
            background_color (str, optional): Color of the background in the render. Defaults to 'black'.
            mode (np.array, optional): An array with shape (n_atoms, 3) representing vectors to be drawn from the center of each atom. Defaults to None.

        Returns:
            pv.Plotter: The PyVista plotter object with the rendered visualization.
        """

        import tkinter as tk
        from tkinter import filedialog

        if atoms_settings is None:
            atoms_settings = DEFAULT_ATOMS_SETTINGS

        if plotter is None:
            if save:
                plotter = pv.Plotter(notebook=False, off_screen=True,
                                     line_smoothing=True, polygon_smoothing=True, image_scale=5)
            else:
                plotter = pv.Plotter(notebook=notebook)

        plotter.set_background(background_color)

        if light_settings:
            plotter.enable_lightkit()

            # Extract the settings from the provided dictionary and update the lights
            for light in plotter.renderer.lights:
                if 'light_position' in light_settings:
                    light.position = light_settings['light_position']
                if 'light_color' in light_settings:
                    light.color = light_settings['light_color']
                if 'light_intensity' in light_settings:
                    light.intensity = light_settings['light_intensity']
                if 'light_type' in light_settings:
                    light.light_type = light_settings['light_type']

        # Get unique symbols from atoms
        symb = list(set(self.symbols))
        _atoms_settings = atoms_settings.loc[symb]

        for position, symbol in zip(self.positions, self.get_chemical_symbols()):
            if symbol in _atoms_settings.index:
                settings = _atoms_settings.loc[symbol]
            else:
                print(
                    f"Warning: {symbol} not found in settings. Using default.")
                settings = _atoms_settings.loc['Unknown']

            if show_hydrogens or symbol != 'H':
                sphere = pv.Sphere(radius=settings['Radius'], center=position,
                                   theta_resolution=resolution, phi_resolution=resolution)
                color = settings['Color']
                # Convert color to RGB array if it's not already
                if isinstance(color, str):
                    color = hex_to_rgb(color)
                rgba_color = np.append(color, alpha)
                rgba_array = np.tile(rgba_color, (sphere.n_points, 1))
                sphere.point_data['RGBA'] = rgba_array.astype(np.uint8)
                plotter.add_mesh(sphere, color=color,
                                 smooth_shading=True, opacity=alpha)

        # Display atom numbers if required
        if atom_numbers:
            poly = pv.PolyData(self.positions)
            poly["My Labels"] = [str(i) for i in range(poly.n_points)]
            plotter.add_point_labels(
                poly, "My Labels", point_size=20, font_size=36)

        # Display hydrogen bonds if required
        if show_hydrogens and show_hydrogen_bonds:
            h_indices = np.where(
                np.array(self.get_chemical_symbols()) == 'H')[0]
            f_indices = np.where(
                np.array(self.get_chemical_symbols()) == 'F')[0]
            for h_index in h_indices:
                h_coord = self.positions[h_index]
                for f_index in f_indices:
                    f_coord = self.positions[f_index]
                    if np.linalg.norm(h_coord - f_coord) < 2.2:
                        line = pv.Line(h_coord, f_coord)
                        plotter.add_mesh(line, color='white',
                                         opacity=alpha, line_width=2)

        # Render bonds
        bonds = self.get_all_bonds()
        for bond in bonds:
            if not show_hydrogens and 'H' in self.G.nodes[bond].get('label', ''):
                continue
            atom_a = self.positions[bond[0]]
            atom_b = self.positions[bond[1]]
            atom_a_radius = _atoms_settings.loc[self.get_chemical_symbols()[
                bond[0]]]['Radius']
            atom_b_radius = _atoms_settings.loc[self.get_chemical_symbols()[
                bond[1]]]['Radius']

            bond_type = self.G[bond[0]][bond[1]].get('bond_type', 0)
            bond_type = 1 if not valency else bond_type

            bond_vector = atom_b - atom_a
            bond_length = np.linalg.norm(bond_vector)
            unit_vector = bond_vector / bond_length

            # Determine orthogonal direction based on the camera position if provided
            if cpos is not None:
                camera_position = np.array(cpos[0])
                focal_point = np.array(cpos[1])
                view_direction = focal_point - camera_position
                orthogonal = np.cross(bond_vector, view_direction)
                # Normalize to make it unit vector
                orthogonal = orthogonal / np.linalg.norm(orthogonal)
            else:
                basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                smallest = np.argmin(np.abs(bond_vector))
                orthogonal = np.cross(bond_vector, basis_vectors[smallest])
                orthogonal = orthogonal / np.linalg.norm(orthogonal)

            match bond_type:
                case 1:
                    # Single bond
                    cylinder_radius = 0.05
                    adjusted_atom_a_radius = np.sqrt(
                        atom_a_radius**2 - cylinder_radius**2)
                    adjusted_atom_b_radius = np.sqrt(
                        atom_b_radius**2 - cylinder_radius**2)
                    atom_a_adjusted = atom_a + unit_vector * adjusted_atom_a_radius
                    atom_b_adjusted = atom_b - unit_vector * adjusted_atom_b_radius
                    bond_length_adjusted = bond_length - \
                        adjusted_atom_a_radius - adjusted_atom_b_radius
                    cylinder = pv.Cylinder(center=0.5*(atom_a_adjusted + atom_b_adjusted),
                                           direction=bond_vector,
                                           height=bond_length_adjusted,
                                           radius=cylinder_radius,
                                           resolution=resolution, capping=False)
                    color = hex_to_rgb('#D3D3D3')
                    rgba_color = np.append(color, alpha)
                    rgba_array = np.tile(rgba_color, (cylinder.n_points, 1))
                    cylinder.point_data['RGBA'] = rgba_array.astype(np.uint8)
                    plotter.add_mesh(cylinder, color='#D3D3D3',
                                     smooth_shading=True, opacity=alpha)

                case 2:
                    # Double bond
                    cylinder_radius = 0.025
                    offset_distance = 0.03
                    for i in [-1, 1]:
                        offset_vector = i * offset_distance * \
                            np.array(orthogonal)
                        offset_magnitude = np.linalg.norm(offset_vector)
                        L_a = adjust_bond_length(
                            atom_a_radius, offset_magnitude, cylinder_radius)
                        L_b = adjust_bond_length(
                            atom_b_radius, offset_magnitude, cylinder_radius)
                        atom_a_adjusted = atom_a + unit_vector * L_a
                        atom_b_adjusted = atom_b - unit_vector * L_b
                        bond_length_adjusted = bond_length + L_a + L_b-atom_a_radius-atom_b_radius
                        cylinder = pv.Cylinder(center=0.5*(atom_a_adjusted + atom_b_adjusted) + offset_vector,
                                               direction=bond_vector,
                                               height=bond_length_adjusted,
                                               radius=cylinder_radius,
                                               resolution=resolution, capping=False)

                        color = hex_to_rgb('#D3D3D3')
                        rgba_color = np.append(color, alpha)
                        rgba_array = np.tile(
                            rgba_color, (cylinder.n_points, 1))
                        cylinder.point_data['RGBA'] = rgba_array.astype(
                            np.uint8)
                        plotter.add_mesh(
                            cylinder, color='#D3D3D3', smooth_shading=True, opacity=alpha)

                case 3:
                    # Triple bond
                    cylinder_radius = 0.02
                    offset_distance = 0.05
                    for i in [-1, 0, 1]:
                        offset_vector = i * offset_distance * \
                            np.array(orthogonal)
                        offset_magnitude = np.linalg.norm(offset_vector)
                        L_a = adjust_bond_length(
                            atom_a_radius, offset_magnitude, cylinder_radius)
                        L_b = adjust_bond_length(
                            atom_b_radius, offset_magnitude, cylinder_radius)
                        atom_a_adjusted = atom_a + unit_vector * L_a
                        atom_b_adjusted = atom_b - unit_vector * L_b
                        bond_length_adjusted = bond_length + L_a + L_b-atom_a_radius-atom_b_radius
                        cylinder = pv.Cylinder(center=0.5*(atom_a_adjusted + atom_b_adjusted) + offset_vector,
                                               direction=bond_vector,
                                               height=bond_length_adjusted,
                                               radius=cylinder_radius,
                                               resolution=resolution, capping=False)
                        color = hex_to_rgb('#D3D3D3')
                        rgba_color = np.append(color, alpha)
                        rgba_array = np.tile(
                            rgba_color, (cylinder.n_points, 1))
                        cylinder.point_data['RGBA'] = rgba_array.astype(
                            np.uint8)
                        plotter.add_mesh(
                            cylinder, color='#D3D3D3', smooth_shading=True, opacity=alpha)

                case _:
                    # Error case
                    cylinder_radius = 0.05
                    adjusted_atom_a_radius = np.sqrt(
                        atom_a_radius**2 - cylinder_radius**2)
                    adjusted_atom_b_radius = np.sqrt(
                        atom_b_radius**2 - cylinder_radius**2)
                    atom_a_adjusted = atom_a + unit_vector * adjusted_atom_a_radius
                    atom_b_adjusted = atom_b - unit_vector * adjusted_atom_b_radius
                    bond_length_adjusted = bond_length - \
                        adjusted_atom_a_radius - adjusted_atom_b_radius
                    cylinder = pv.Cylinder(center=0.5*(atom_a_adjusted + atom_b_adjusted),
                                           direction=bond_vector,
                                           height=bond_length_adjusted,
                                           radius=cylinder_radius,
                                           resolution=resolution, capping=False)
                    color = hex_to_rgb('#FF0000')
                    rgba_color = np.append(color, alpha)
                    rgba_array = np.tile(rgba_color, (cylinder.n_points, 1))
                    cylinder.point_data['RGBA'] = rgba_array.astype(np.uint8)
                    plotter.add_mesh(cylinder, color='#FF0000',
                                     smooth_shading=True, opacity=alpha)

        if scalar_fields_settings is not None:
            def render_scalar_field(scalar_field, settings):
                _settings = settings.copy()
                _settings['plotter'] = plotter
                _settings['save'] = False
                _settings['show'] = False
                _settings['notebook'] = notebook
                scalar_field.render(**_settings)

            assert isinstance(
                scalar_fields_settings, dict), "scalar_fields_settings must be a dictionary."
            if len(self.scalar_fields) > 0:
                if len(scalar_fields_settings) > 0 and set(scalar_fields_settings.keys()).issubset(set(self.scalar_fields.keys())):
                    # Individual settings for each scalar field
                    print("Rendering individual scalar fields.")
                    print(f"{scalar_fields_settings.keys()=}")
                    print(f'{self.scalar_fields.keys()=}')
                    for scalar_field_name, scalar_field_settings in scalar_fields_settings.items():
                        render_scalar_field(
                            self.scalar_fields[scalar_field_name], scalar_field_settings)
                else:
                    # Default settings for all scalar fields
                    for scalar_field in self.scalar_fields.values():
                        render_scalar_field(
                            scalar_field, scalar_fields_settings)
            else:
                warnings.warn('There are no scalar fields to render.')

        if show_basis_vectors:
            origin = np.array([0, 0, 0])

            basis_vectors = [np.array([1, 0, 0]), np.array(
                [0, 1, 0]), np.array([0, 0, 1])]
            colors = ['red', 'green', 'blue']

            for direction, color in zip(basis_vectors, colors):
                arrow = pv.Arrow(start=origin, direction=direction,
                                 shaft_radius=0.05, tip_radius=0.1)
                plotter.add_mesh(arrow, color=color)

        if mode is not None:
            # Validate the mode argument
            if not isinstance(mode, np.ndarray) or mode.shape != (len(self), 3):
                raise ValueError(
                    "mode must be a numpy array with shape (n_atoms, 3)")

            # Get positions of atoms
            positions = self.get_positions()

            # Iterate over atoms and add vectors
            for i, position in enumerate(positions):
                start_point = position
                end_point = position + mode[i]
                arrow = pv.Arrow(start=start_point,
                                 direction=end_point - start_point)
                # You can choose the color that suits your visualization
                plotter.add_mesh(arrow, color='purple')

        # Display atom IDs if required
        if show_numbers:
            poly = pv.PolyData(self.positions)
            poly["Atom IDs"] = [str(atom.index) for atom in self]
            plotter.add_point_labels(
                poly, "Atom IDs", point_size=20, font_size=36, render_points_as_spheres=False)

        # If saving is required, save the screenshot
        if isinstance(save, str):
            plotter.show(window_size=[1000, 1000], cpos=cpos)
            plotter.screenshot(save)
            return None

        if save_3d:
            combined_mesh = pv.PolyData()

            for actor in plotter.renderer.actors.values():
                if hasattr(actor, 'GetMapper'):
                    mesh = actor.GetMapper().GetInputAsDataSet()
                    if mesh.is_all_triangles:
                        mesh = pv.wrap(mesh.extract_surface())

                    if 'RGBA' in mesh.point_data:
                        colors = mesh.point_data['RGBA']
                        assert len(
                            colors) == mesh.n_points, "Mismatch in the number of color points and mesh points."

                        # Assign colors to mesh
                        mesh.point_data['RGBA'] = colors.astype(np.uint8)

                        # Combine the colored mesh
                        combined_mesh += mesh

            if combined_mesh.n_points > 0:
                # Save the combined mesh with color information as a texture
                combined_mesh.save(save_3d, texture='RGBA')
            else:
                warnings.warn(
                    "No valid color data to combine, or combined_mesh is empty.")

        # If showing is required, display the visualization
        if show:
            # Hotkey functions
            def copy_camera_position_to_clipboard():
                cam_pos = plotter.camera_position
                cam_pos_str = f"{cam_pos}"
                print(cam_pos_str)
                pyperclip.copy(cam_pos_str)

            def save_render_view_with_dialog():
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
                                                        ("PNG files", "*.png"), ("All files", "*.*")])
                if file_path:
                    plotter.screenshot(file_path)
                    print(f"Saved image as '{file_path}'.")

            plotter.add_key_event("c", copy_camera_position_to_clipboard)
            plotter.add_key_event("p", save_render_view_with_dialog)

            plotter.show(window_size=[1000, 1000], cpos=cpos,
                         auto_close=auto_close, interactive=interactive)

        return plotter

    def render_alongside(self, other, other_alpha=1.0, *args, **kwargs):
        # Store the original value of 'show', defaulting to True if not provided
        original_show_value = kwargs.get('show', True)
        original_save_value = kwargs.get('save', False)
        # Set 'show' to False for the first render
        kwargs['show'] = False
        # if was string, will make argumet to True, to go to screeenshot mode
        kwargs['save'] = bool(original_save_value)
        p = self.render(*args, **kwargs)
        # Restore the original value of 'show' for the second render
        kwargs['show'] = original_show_value
        kwargs['save'] = original_save_value
        # Update 'alpha' value for the second render if not provided
        kwargs['alpha'] = kwargs.get('alpha', other_alpha)
        return other.render(p, *args, **kwargs)

    def scheme(self, filename=None):
        """Compute 2D coordinates for the molecule using RDKit and visualize or save it."""
        mol = self.to_rdkit_mol()

        # Compute the 2D coordinates
        Compute2DCoords(mol)

        # If a filename is provided, save the image. Otherwise, display it in Jupyter.
        if filename:
            Draw.MolToFile(mol, filename)
        else:
            pil_img = Draw.MolToImage(mol)
            byte_stream = BytesIO()
            pil_img.save(byte_stream, format="PNG")
            display(Image(data=byte_stream.getvalue()))

    def get_fragment(self, fragment_attach_atom, fragment_atoms_ids):
        """
        Function to get fragment from a molecule
        fragment_attach_atom - atom in fragment that is attached to the rest of the molecule
        fragment_atoms_ids - ids of the atoms fragments
        """
        _, fragment = self.divide_in_two_fragments(
            fragment_attach_atom, fragment_atoms_ids)
        return fragment

    def to_xyz_string(self):
        sio = io.StringIO()
        write(sio, self, format="xyz")
        return sio.getvalue()

    def save(self, filename):
        with open(filename, "w") as text_file:
            text_file.write(self.to_xyz_string())

    def _to_cacheable_format(self):
        """Convert the Motor object into a cacheable format based on atomic positions and symbols."""
        return (tuple(self.get_positions().flatten()), tuple(self.get_chemical_symbols()))

    def linear_interpolation(self, atoms: Atoms, n: int = 1):
        from .path import Path
        path = Path([self])
        for i in range(n):
            new_atmos = self.copy()
            new_atmos.set_positions(self.get_positions(
            )+((i+1)/(n+1))*(atoms.get_positions()-self.get_positions()))
            path.append(new_atmos)
        return path
