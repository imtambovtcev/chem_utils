from __future__ import annotations

import io
import math
import pathlib
import tkinter as tk
import warnings
from collections import Counter
from io import BytesIO
from itertools import islice
from tkinter import filedialog

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

from .valency import add_valency, general_print, rebond

default_atoms_settings = pd.read_csv(
    str(pathlib.Path(__file__).parent/'pyvista_render_settings.csv'))
default_atoms_settings.index = list(default_atoms_settings['Name'])
default_atoms_settings['Color'] = [[int(i) for i in s.replace(
    '[', '').replace(']', '').split(',')] for s in default_atoms_settings['Color']]


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

        # print(f'{unique_bonds = }')
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
            assert bond_count <= atom_valency, f"Atom {node} ({atom_symbol}) has {bond_count} bonds, which exceeds its known valency of {atom_valency}!"

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
        # Use ASE's read function to get Atoms object from file
        atoms = read(filename)

        # Create a Molecule instance from the Atoms object
        molecule = cls(atoms)

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

        # print(f'{self_counts = }')
        # print(f'{other_counts = }')

        # Check if the atom type counts are the same
        return self_counts == other_counts

    def copy(self):
        """Return a copy of the Molecule."""
        new_molecule = Molecule()
        for key, value in self.arrays.items():
            new_molecule.arrays[key] = value.copy()
        new_molecule.set_cell(self.get_cell().copy())
        new_molecule.set_pbc(self.get_pbc().copy())
        new_molecule.info = self.info.copy()
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
            raise ValueError("mode must be a numpy array with shape (n_atoms, 3)")
            
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

    def divide_in_two_fragments(self, fragment_attach_atom, fragment_atoms_ids):
        """
        fragment_attach_atom - atom in fragment that is attached to the rest of the molecule
        fragment_atoms_ids - ids of the atoms fragments
        """
        from .fragment import Fragment
        small_fragment_ids = list(set(fragment_atoms_ids))
        # print(f'{small_fragment_ids = }')
        main_fragment_ids = list(
            (set(range(len(self)))-set(small_fragment_ids)) | {fragment_attach_atom})
        # print(f'{self.get_bonds_of(small_fragment_ids) = }')
        small_fragment_bonds = [bond for bond in self.get_bonds_of(
            small_fragment_ids) if set(bond).issubset(set(small_fragment_ids))]
        # print(f'{small_fragment_bonds = }')
        main_fragment_bonds = [
            bond for bond in self.get_all_bonds() if bond not in small_fragment_bonds]
        # print(f'{main_fragment_bonds = }')
        new_small_fragment_ids = dict(
            zip(small_fragment_ids, range(len(small_fragment_ids))))
        # print(f'{new_small_fragment_ids = }')
        new_main_fragment_ids = dict(
            zip(main_fragment_ids, range(len(main_fragment_ids))))
        # print(f'{new_main_fragment_ids = }')
        V0_s, V1_s, V2_s, V3_s = fragment_vectors(
            new_small_fragment_ids[fragment_attach_atom], self.get_positions()[small_fragment_ids, :])
        V0_m, V1_m, V2_m, V3_m = fragment_vectors(
            new_main_fragment_ids[fragment_attach_atom], self.get_positions()[main_fragment_ids, :])
        # print(f'{set(self.get_chemical_symbols())[small_fragment_ids] = }')

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
                return None
            all_iso = nx.vf2pp_all_isomorphisms(
                self.G, other.G, node_label='label')
        else:
            if not nx.is_isomorphic(self.G, other.G):
                return None
            all_iso = nx.vf2pp_all_isomorphisms(self.G, other.G)

        return all_iso

    def iso_distance(self, other, iso=None):
        coord_1 = self.get_positions()
        coord_2 = other.get_positions()
        if iso is not None:
            coord_1 = coord_1[np.array(list(iso.keys()), dtype=int), :]
            coord_2 = coord_2[np.array(list(iso.values()), dtype=int), :]
        # print(np.linalg.norm(cdist(coord_1, coord_1) - cdist(coord_2, coord_2)))
        return np.linalg.norm(cdist(coord_1, coord_1) - cdist(coord_2, coord_2))

    def best_mapping(self, other, limit=None):
        if not (self.is_isomorphic(other)):
            return None
        isos = list(islice(self.all_mappings(other), limit))
        return min(isos, key=lambda x: self.iso_distance(other, x))

    def best_distance(self, other: Atoms, limit=None):
        if self.is_isomorphic(other):
            new = other.reorder(self.best_mapping(other, limit=limit))
            return self.iso_distance(new)
        else:
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

    def rotate_part(self, atom, bond, angle):
        edge = bond if bond[0] == atom else bond[::-1]
        r0 = self.get_positions()[atom]
        v = self.get_positions()[edge[1]]-r0
        island = self.divide_in_two(edge)[0]
        r = rotation_matrix(v, np.pi*angle/180)
        new_coords = np.matmul(r, (self.get_positions()[island]-r0).T).T+r0
        self.set_positions(new_coords, island)
        return self

    def render(self, plotter: pv.Plotter = None, show=False, save=None, atoms_settings=default_atoms_settings, show_hydrogens=True, alpha=1.0, atom_numbers=False, show_hydrogen_bonds=False, show_numbers=False, show_basis_vectors=False, cpos=None, notebook=False, auto_close=True, interactive=True, background_color='black', valency=False, resolution=20,  light_settings=None, mode=None):
        """
        Renders a 3D visualization of a molecule using the given settings.

        Args:
            plotter (pv.Plotter, optional): The PyVista plotter object used for rendering. If not provided, a new one will be created. Defaults to None.
            atoms_settings (DataFrame, optional): A dataframe containing the visualization settings for each atom type. Defaults to default_atoms_settings.
            show_hydrogens (bool, optional): Whether to render hydrogen atoms. Defaults to True.
            alpha (float, optional): The opacity of the rendered atoms and bonds. Value should be between 0 (transparent) and 1 (opaque). Defaults to 1.0.
            atom_numbers (bool, optional): Whether to display atom numbers. Defaults to False.
            show_hydrogen_bonds (bool, optional): Whether to visualize hydrogen bonds. Defaults to False.
            show_numbers (bool, optional): Whether to show atom indices. Defaults to False.
            show_basis_vectors (bool, optional): Whether to display the basis vectors. Defaults to True.
            save (str or bool, optional): Filepath to save the rendered image. If provided, the rendered image will be saved to this path, if bool True and no plotter provided, will create a plotter for screenshot. Defaults to None.
            cpos (list or tuple, optional): Camera position for the PyVista plotter. Defaults to None.
            notebook (bool, optional): Whether the function is being called within a Jupyter notebook. Adjusts the rendering accordingly. Defaults to False.
            auto_close (bool, optional): Whether to automatically close the rendering window after saving or completing the rendering. Defaults to True.
            interactive (bool, optional): Whether to allow interactive visualization (e.g., rotation, zoom). Defaults to False.
            background_color (str, optional): Color of the background in the render. Defaults to 'black'.
            mode (np.array, optional): An array with shape (n_atoms, 3) representing vectors to be drawn from the center of each atom. Defaults to None.

        Returns:
            pv.Plotter: The PyVista plotter object with the rendered visualization.
        """

        if atoms_settings is None:
            atoms_settings = default_atoms_settings

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
            # Ensure symbol is in the DataFrame's index
            if symbol in _atoms_settings.index:
                settings = _atoms_settings.loc[symbol]
            else:
                print(
                    f"Warning: {symbol} not found in settings. Using default.")
                settings = _atoms_settings.loc['Unknown']

            if show_hydrogens or symbol != 'H':
                sphere = pv.Sphere(
                    radius=settings['Radius'], center=position, theta_resolution=resolution, phi_resolution=resolution)
                plotter.add_mesh(
                    sphere, color=settings['Color'], smooth_shading=True, opacity=alpha)

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
                    plotter.add_mesh(cylinder, color='#FF0000',
                                     smooth_shading=True, opacity=alpha)

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

        # If saving is required, save the screenshot
        if isinstance(save, str):
            plotter.show(window_size=[1000, 1000], cpos=cpos)
            plotter.screenshot(save)
            return None

        plotter.add_key_event("c", copy_camera_position_to_clipboard)
        plotter.add_key_event("p", save_render_view_with_dialog)

        # If showing is required, display the visualization
        if show:
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

    def rotate(self, zero_atom=None, x_atom=None, no_z_atom=None):
        """
        rotate molecule by 3 atoms:
        - zero_atom: to be placed to (0,0,0)
        - x_atom: to be placed to (?, 0, 0)
        - no_z_atom: to be placed to (?, ?, 0)
        """
        if all(variable is None for variable in (zero_atom, x_atom, no_z_atom)):
            zero_atom, x_atom, no_z_atom = self.find_rotation_atoms()
        positions = self.get_positions()
        positions -= positions[zero_atom, :]
        positions = np.matmul(rotation_matrix(np.cross(positions[x_atom, :], [1, 0, 0]), np.arccos(
            np.dot(positions[x_atom, :], [1, 0, 0])/np.linalg.norm(positions[x_atom, :]))), positions.T).T
        positions = np.matmul(rotation_matrix(np.array(
            [1., 0., 0.]), -np.arctan2(positions[no_z_atom, 2], positions[no_z_atom, 1])), positions.T).T
        self.set_positions(positions)

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

    def linear_interploation(self, atoms: Atoms, n=1):
        from .path import Path
        path = Path([self])
        for i in range(n):
            new_atmos = self.copy()
            new_atmos.set_positions(self.get_positions(
            )+((i+1)/(n+1))*(atoms.get_positions()-self.get_positions()))
            path.append(new_atmos)
        return path
