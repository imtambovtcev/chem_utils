from __future__ import annotations

import io
import math
import tkinter as tk
import warnings
from collections import Counter
from io import BytesIO
from itertools import combinations, islice
from pathlib import Path
from tkinter import filedialog
from warnings import warn

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
import pyperclip
import pyvista as pv
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.io import read, write
from IPython.display import Image, display
from rdkit import Chem
from rdkit.Chem import (AddHs, AllChem, CanonicalRankAtoms, Draw, GetSSSR,
                        SanitizeMol, rdFMCS)
from rdkit.Chem.rdDepictor import Compute2DCoords
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

from .valency import add_valency, general_print, rebond

default_atoms_settings = pd.read_csv(
    str(Path(__file__).parent/'pyvista_render_settings.csv'))
default_atoms_settings.index = list(default_atoms_settings['Name'])
default_atoms_settings['Color'] = [[int(i) for i in s.replace(
    '[', '').replace(']', '').split(',')] for s in default_atoms_settings['Color']]


def adjust_bond_length(atom_radius, offset_distance, cylinder_radius):
    # Calculate additional length for atom
    x = abs(offset_distance) + cylinder_radius
    L = atom_radius-np.sqrt(atom_radius**2 - x**2)
    return L


def add_vector_to_plotter(p, r0, v, color='blue'):
    # Create a single point from the origin
    points = pv.PolyData(r0)

    # Associate the orientation vector with the dataset
    # Making sure the vector is a 2D array with shape (n_points, 3)
    points['Vectors'] = np.array([v])

    # Generate arrows
    arrows = points.glyph(orient='Vectors', scale=False,
                          factor=1.0, geom=pv.Arrow())

    # Add arrows to the plotter
    p.add_mesh(arrows, color=color)


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

    def divide_in_two(self, bond):
        G = self.G.copy()
        G.remove_edge(bond[0], bond[1])
        return list(nx.shortest_path(G, bond[0]).keys()), list(nx.shortest_path(G, bond[1]).keys())

    def divide_in_two_fragments(self, fragment_attach_atom, fragment_atoms_ids):
        """
        fragment_attach_atom - atom in fragment that is attached to the rest of the molecule
        fragment_atoms_ids - ids of the atoms fragments
        """
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

    def render(self, plotter: pv.Plotter = None, show=False, save=None, atoms_settings=default_atoms_settings, show_hydrogens=True, alpha=1.0, atom_numbers=False, show_hydrogen_bonds=False, show_numbers=False, show_basis_vectors=False, cpos=None, notebook=False, auto_close=True, interactive=True, background_color='black', valency=False, resolution=20,  light_settings=None):
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

    def scheme_2D(self, filename=None):
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
        path = Path([self])
        for i in range(n):
            new_atmos = self.copy()
            new_atmos.set_positions(self.get_positions(
            )+((i+1)/(n+1))*(atoms.get_positions()-self.get_positions()))
            path.append(new_atmos)
        return path


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

    def fragment_vectors_render(self, plotter: pv.Plotter = None, notebook=False):
        if plotter is None:
            plotter = pv.Plotter(notebook=notebook)
        V0, V1, V2, V3 = self.fragment_vectors
        add_vector_to_plotter(plotter, V0, V1, color='red')
        add_vector_to_plotter(plotter, V0, V2, color='green')
        add_vector_to_plotter(plotter, V0, V3, color='blue')
        return plotter

    def apply_transition(self, r0):
        self.set_positions(self.get_positions() + r0)

    def apply_rotation(self, R):
        # Rotate atom positions (assuming each row is a position vector)
        self.set_positions(np.dot(R, self.get_positions().T).T)

        # Rotate the attach_matrix (assuming each column is a vector)
        self.attach_matrix = np.dot(R, self.attach_matrix)

    def get_origin_rotation_matrix(self):
        V0, V1, V2, V3 = self.fragment_vectors
        return np.linalg.inv(np.column_stack([V1, V2, V3]))

    def set_to_origin(self):
        self.apply_transition(-self.attach_point)
        self.apply_rotation(self.get_origin_rotation_matrix())

    def render(self, **kwargs):
        if 'plotter' in kwargs:
            plotter = self.fragment_vectors_render(kwargs['plotter'])
        else:
            notebook=kwargs.get('Notebook',False)
            plotter = self.fragment_vectors_render(None, notebook=notebook)

        # Update the plotter in kwargs before calling super
        kwargs['plotter'] = plotter
        return super().render(**kwargs)

    def render_alongside(self, other, other_alpha=1.0, **kwargs):
        # Get the plotter from kwargs or default to None
        plotter = kwargs.get('plotter', None)

        # Process the plotter using fragment_vectors_render
        plotter = self.fragment_vectors_render(plotter)

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
                    f'{len(molecule.get_all_bonds()) = } {len(self.get_all_bonds()) + len(_fragment.get_all_bonds()) = }')
                scale_factor += 0.01
                print(f'{scale_factor = }')
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

    # def __add__(self, fragment):
    #     return self.connect_fragment(fragment)


# Define the standard_stator graph
standard_stators = [[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 8), (4, 6), (4, 9), (5, 10), (6, 12), (8, 14), (9, 14), (10, 16), (12, 16)],
                    [(3, 7), (5, 7), (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 8), (4, 10),
                     (5, 11), (6, 12), (8, 14), (10, 14), (11, 18), (12, 18), (7, 3), (7, 5)],
                    [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 8), (4, 9), (4, 10), (5, 11), (6, 13), (6, 14), (8, 16), (9, 16), (10, 14), (11, 19), (13, 19)]]

standard_stators_list = []

for ss in standard_stators:
    standard_stator = nx.Graph()
    standard_stator.add_edges_from(ss)
    standard_stators_list.append(standard_stator)


class Motor(Molecule):
    def __init__(self, atoms):
        super().__init__(atoms)

    def find_bond(self):
        """
        Identify the bond connecting the stator and rotor in a molecule.
        Returns:
        - dict: Information about the bond, stator, and rotor nodes and their neighbors.
        bond[0] -- stator
        bond[1] -- rotor
        """

        edges = self.get_bonds_without()
        # Create a directed graph from the provided edges

        # print(f'{edges = }')

        Gu = nx.Graph(edges)
        G = Gu.copy().to_directed()

        # Find all simple cycles and sort them in descending order based on length
        cycles = sorted(nx.simple_cycles(G), key=len, reverse=True)

        # Extract cycles that don't share nodes with the largest cycle
        small_cycle = [c for c in cycles if not set(c) & set(cycles[0])]

        # Identify the bond connecting the stator and rotor
        bond = None
        for a in small_cycle[0]:
            b = [j for _, j in G.out_edges(a) if j in cycles[0]]
            if b:
                bond = [b[0], a]
                break

        if bond is None:
            raise ValueError("Bond connecting stator and rotor not found.")

        Gu.remove_edge(bond[0], bond[1])

        part1, part2 = list(nx.shortest_path(Gu, bond[0]).keys()), list(
            nx.shortest_path(Gu, bond[1]).keys())

        part1 = Gu.subgraph(part1).copy()
        part2 = Gu.subgraph(part2).copy()

        # print(f'{part1.nodes = }')
        # print(f'{part2.nodes = }')
        # Identify which part is stator and which is rotor :
        if any(nx.is_isomorphic(part1, standard_stator) for standard_stator in standard_stators_list):
            pass

        elif any(nx.is_isomorphic(part2, standard_stator) for standard_stator in standard_stators_list):
            bond.reverse()

        else:  # both parts are not stator! Simple stator finding procedure:
            print('No stator isomorphysm found!')
            # Classify nodes based on 6-membered rings
            flat_cycles_6 = {item for i in cycles if len(i) == 6 for item in i}
            # If presumed stator neighbors are not in 6-membered rings, switch stator and rotor classifications
            stator_neighbours = sorted(
                [j for _, j in G.out_edges(bond[0]) if j != bond[1]])
            print(f'{stator_neighbours = }')
            if not all(neighbor in flat_cycles_6 for neighbor in stator_neighbours):
                bond.reverse()

        stator_neighbours = sorted(
            [j for _, j in G.out_edges(bond[0]) if j != bond[1]])
        rotor_neighbours = sorted(
            [j for _, j in G.out_edges(bond[1]) if j != bond[0]])

        # Return the identified bond, nodes, and their neighbors
        return {
            'bond': bond,
            'bond_stator_node': bond[0],
            'bond_rotor_node': bond[1],
            'stator_neighbours': stator_neighbours,
            'rotor_neighbours': rotor_neighbours
        }

    def get_stator_rotor_bond(self):
        # bonds = [item for x in set(self.get_chemical_symbols()) if x !='H' for item in self.ana.get_bonds('C', x, unique=True)[0]]+ [item for x in set(self.get_chemical_symbols()) if x !='H' for item in self.ana.get_bonds('S', x, unique=True)[0]]+ [item for x in set(self.get_chemical_symbols()) if x !='H' for item in self.ana.get_bonds('O', x, unique=True)[0]]
        f_b = self.find_bond()
        # 'H' or 'F' bonded to this atom
        if len(set(('H', 'F')) & set(self.id_to_symbol(self.get_bonded_atoms_of(f_b['rotor_neighbours'][0])))) > 0:
            f_b['C_H_rotor'] = f_b['rotor_neighbours'][0]
            f_b['not_C_H_rotor'] = f_b['rotor_neighbours'][1]
        else:
            f_b['C_H_rotor'] = f_b['rotor_neighbours'][1]
            f_b['not_C_H_rotor'] = f_b['rotor_neighbours'][0]
        if self.spatial_distance_between_atoms(f_b['C_H_rotor'], f_b['stator_neighbours'][0]) < self.spatial_distance_between_atoms(f_b['C_H_rotor'], f_b['stator_neighbours'][1]):
            f_b['C_H_stator'] = f_b['stator_neighbours'][0]
            f_b['not_C_H_stator'] = f_b['stator_neighbours'][1]
        else:
            f_b['C_H_stator'] = f_b['stator_neighbours'][1]
            f_b['not_C_H_stator'] = f_b['stator_neighbours'][0]
        return f_b

    def get_stator_rotor(self):
        f_b = self.get_stator_rotor_bond()
        stator, rotor = self.divide_in_two(f_b['bond'])
        stator = self.get_fragment(f_b['bond_stator_node'], stator)
        rotor = self.get_fragment(f_b['bond_rotor_node'], rotor)
        return stator, rotor

    def get_rotor_H(self):
        f_b = self.get_stator_rotor_bond()
        # for i in self.get_bonded_atoms_of(f_b['C_H_stator']):
        #     print(i)
        #     if len(set(('H','F')) & set(self.id_to_symbol([i])))>0:
        #         print('Yes')
        return [i for i in self.get_bonded_atoms_of(f_b['C_H_rotor']) if len(set(('H', 'F')) & set(self.id_to_symbol([i]))) > 0][0]

    def get_stator_N(self):
        f_b = self.get_stator_rotor_bond()
        H = self.G.copy()
        H.remove_node(f_b['C_H_stator'])
        l = self.get_bonded_atoms_of(f_b['C_H_stator'])
        l.remove(f_b['bond_stator_node'])
        if nx.shortest_path_length(H, l[0], f_b['bond_stator_node']) > nx.shortest_path_length(H, l[1], f_b['bond_stator_node']):
            return l[0]
        else:
            return l[1]

    def get_stator_H(self):
        l = [i for i in self.get_bonded_atoms_of(self.get_stator_N()) if len(
            set(('H', 'F')) & set(self.id_to_symbol([i]))) > 0]
        return l[0] if len(l) > 0 else None

    def get_break_bonds(self):
        bonds = self.get_bonds_without()
        break_bond = []
        connecting_bond = self.get_stator_rotor_bond()['bond']
        for bond in bonds:
            if set(connecting_bond) != set(bond):
                G = nx.Graph(bonds)
                G.remove_edge(bond[0], bond[1])
                G.remove_nodes_from(list(nx.isolates(G)))

                if not nx.is_connected(G):
                    break_bond.append(bond)
        return break_bond

    def get_break_nodes(self):
        rotor_node = self.get_stator_rotor_bond()['bond_rotor_node']
        return [b[np.argmax([nx.shortest_path_length(self.G, b[0], rotor_node), nx.shortest_path_length(self.G, b[1], rotor_node)])] for b in self.get_break_bonds()]

    def get_tails(self):
        f_b = self.get_stator_rotor_bond()
        b_b = self.get_break_bonds()
        # print(f'{f_b = }')
        # print(f'{b_b = }')

        if any(f_b['C_H_rotor'] in b for b in b_b):
            split_bond = [b for b in b_b if f_b['C_H_rotor'] in b]
            # print(f'{split_bond = }')
            if not split_bond:
                return []  # Return empty list if split_bond is empty
            split_bond = split_bond[0]
            a, b = self.divide_in_two(split_bond)
            tail_fragment = b if f_b['C_H_rotor'] in a else a
            # print(f'{a = }')
            # print(f'{f_b["C_H_rotor"] = }')
            assert f_b['C_H_rotor'] not in tail_fragment
            return [node for node in tail_fragment if self.G.nodes[node]['label'] in ['F', 'H']]

        else:
            for i in self.G[f_b['C_H_rotor']]:
                tail_C = i
                tail_H = [j for j in list(self.G[tail_C]) if self.id_to_symbol(
                    j) == 'H' or self.id_to_symbol(j) == 'F']
                if len(tail_H) == 3:
                    return tail_H
            else:  # This 'else' corresponds to the inner 'for' loop
                if 'split_bond' not in locals():  # Ensure split_bond is defined
                    return []  # Return empty list if no condition is met

                tail_start = list(split_bond)
                tail_start.remove(f_b['C_H_rotor'])
                tail_start = tail_start[0]
                sp = self.divide_in_two((f_b['C_H_rotor'], tail_start))
                H = self.G.copy()
                for i in sp[0]:
                    H.remove_node(i)
                tail_distance = nx.shortest_path_length(H, source=tail_start)
                max_distance = np.max(list(tail_distance.values()))

                return [i for i in tail_distance.keys() if tail_distance[i] == max_distance]

    def find_replacement(self, other):
        # Получаем номера замененных вершин
        old_nodes = [self.get_stator_rotor_bond()['bond_stator_node'],
                     self.get_stator_rotor_bond()['bond_rotor_node']]

        # Создаем подграф вокруг этих вершин в исходном графе
        old_subgraph_nodes = set(old_nodes)
        old_subgraph_edges = set()
        for node in old_nodes:
            adj_nodes = list(self.G.adj[node])
            old_subgraph_nodes.update(adj_nodes)
            old_subgraph_edges.update((node, adj_node)
                                      for adj_node in adj_nodes)

        # Получаем номера вершин, на которые были заменены старые вершины
        new_nodes = [other.get_stator_rotor_bond()['bond_stator_node'],
                     other.get_stator_rotor_bond()['bond_rotor_node']]

        # Создаем подграф вокруг этих вершин в получившемся графе
        new_subgraph_nodes = set(new_nodes)
        new_subgraph_edges = set()
        for node in new_nodes:
            adj_nodes = list(other.G.adj[node])
            new_subgraph_nodes.update(adj_nodes)
            new_subgraph_edges.update((node, adj_node)
                                      for adj_node in adj_nodes)

        # Находим различия между подграфами
        different_nodes = new_subgraph_nodes - old_subgraph_nodes
        different_edges = new_subgraph_edges - old_subgraph_edges

        # Возвращаем различия как замену
        return {
            'added_nodes': different_nodes,
            'added_edges': different_edges,
        }

    def find_rotation_atoms(self, settings=None):
        if settings is None or settings['mode'] == 'default':
            f_b = self.get_stator_rotor_bond()
            zero_atom, x_atom, no_z_atom = f_b['bond_rotor_node'], f_b['bond_stator_node'], f_b['C_H_rotor']
        elif settings['mode'] == 'stator':
            f_b = self.get_stator_rotor_bond()
            zero_atom, x_atom, no_z_atom = f_b['bond_stator_node'], f_b[
                'stator_neighbours'][0], f_b['stator_neighbours'][1]
        elif all(key in settings for key in ['zero_atom', 'x_atom', 'no_z_atom']):
            zero_atom, x_atom, no_z_atom = settings['zero_atom'], settings['x_atom'], settings['no_z_atom']
            return zero_atom, x_atom, no_z_atom
        else:
            raise ValueError(
                "Invalid settings provided for finding rotation atoms.")

        return zero_atom, x_atom, no_z_atom

    def reorder(self, mapping: dict):
        """
        Reorders the atoms in the Motor instance based on the provided mapping.

        Parameters:
            - mapping (dict): A dictionary where keys are current indices and values are target indices.

        Returns:
            - Motor: A new Motor instance with atoms reordered based on the mapping.
        """
        reordered_molecule = super().reorder(
            mapping)  # Call the reorder method from the Molecule class
        # Convert the reordered molecule to a Motor instance and return
        return Motor(reordered_molecule)


class Path:
    def __init__(self, images=None):
        self.images = []
        if images is not None:
            # Check if any image is of the Motor class
            contains_motor = any(isinstance(image, Motor) for image in images)
            if contains_motor:
                # Convert all images to Motor class
                converted_images = [Motor(image) if not isinstance(
                    image, Motor) else image for image in images]
            else:
                # Convert all images to Molecule class
                converted_images = [Molecule(image) if not isinstance(
                    image, Molecule) else image for image in images]

            # Check that all images satisfy __eq__ requirements
            reference_image = converted_images[0]
            for idx, img in enumerate(converted_images[1:], start=1):
                if reference_image != img:
                    warn(
                        f"The image at index {idx} atoms are different.")

            # Initialize the self.images attribute
            self.images = converted_images

    def _get_type(self):
        """Retrieve the type of the images in the path."""
        return type(self[0]) if len(self.images) > 0 else None

    def _convert_type(self, image):
        """Convert the image to the type of images in the path."""
        if len(self.images) == 0:
            if not isinstance(image, Molecule):
                return Molecule(image)
            return image
        current_type = self._get_type()
        # print(f'{current_type = }')
        return current_type(image)

    def to_type(self, new_type):
        """
        Convert all images in the path to the specified type.

        Parameters:
        - new_type (type): The desired type to which all images should be converted.
        """
        if not isinstance(new_type, type):
            raise TypeError(
                f"Expected 'new_type' to be a type, got {type(new_type)}")

        self.images = [new_type(image) for image in self.images]

    def __getattr__(self, attr):
        """
        If the attribute (method in this context) is not found, 
        this method is called.
        """
        def method(*args, **kwargs):
            return [getattr(image, attr)(*args, **kwargs) for image in self.images]

        return method

    def __getitem__(self, index):
        """Retrieve an image at the specified index."""
        return self.images[index]

    def __setitem__(self, index, image):
        """Replace an image at the specified index."""

        # Handle single integer index
        if isinstance(index, int):
            converted_image = self._convert_type(image)
            # Check for equality with the first image in the path
            if len(self.images) > 0 and self.images[0] != converted_image:
                warn(
                    f"The provided image does not satisfy the equality requirements with the first image in the path.")
            self.images[index] = converted_image
            return

        # Handle slices
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.images))
            if not isinstance(image, (list, tuple)):
                raise ValueError(
                    "For slice assignment, the image must be a list or tuple of images.")
            for i, img in zip(range(start, stop, step), image):
                self[i] = img
            return

        # Handle lists or other iterables
        if isinstance(index, (list, tuple)):
            if not isinstance(image, (list, tuple)) or len(index) != len(image):
                raise ValueError(
                    "For list or tuple assignment, the image must be a list or tuple of images with the same length as the index.")
            for i, img in zip(index, image):
                self[i] = img
            return

        raise TypeError(f"Index of type {type(index)} is not supported.")

    def __iter__(self):
        """Make the class iterable."""
        return iter(self.images)

    def __len__(self):
        """Return the number of images in the list."""
        return len(self.images)

    def append(self, image):
        """Add a new image to the list."""
        # Convert the image to the appropriate type
        converted_image = self._convert_type(image)

        # Check if the image satisfies __eq__ requirements with the existing images
        if self.images and not all(img == converted_image for img in self.images):
            warn("The appended image does not satisfy the equality requirements with existing images in the path.")

        self.images.append(converted_image)

    def remove_image(self, index):
        """Remove an image from the list at the specified index."""
        del self.images[index]

    def save(self, filename):
        """Save all images in the list to a file."""
        write(filename, self.images)

    @classmethod
    def load(cls, filename):
        """Load images from a file and return a new Path object with those images."""
        return cls(read(filename, index=':'))

    def rotate(self):
        assert isinstance(self[0], Motor)
        zero_atom, x_atom, no_z_atom = self[0].find_rotation_atoms()
        for motor in self:
            motor.rotate(zero_atom, x_atom, no_z_atom)

    def reorder_atoms_of_intermediate_images(self):
        half_len = len(self) // 2

        for i in range(1, half_len + 1):
            mapping = self[i - 1].best_mapping(self[i])
            if mapping is None:
                raise ValueError(
                    f"No valid mapping found for molecules at index {i} and {i-1}.")
            self[i] = self[i].reorder(mapping)

        # Adjust from the end towards the center
        for i in range(len(self) - 2, half_len - 1, -1):
            mapping = self[i + 1].best_mapping(self[i])
            if mapping is None:
                raise ValueError(
                    f"No valid mapping found for molecules at index {i} and {i+1}.")
            self[i] = self[i].reorder(mapping)

    def find_bonds_breaks(self):
        assert self[0].get_all_bonds() == self[-1].get_all_bonds()
        first_image_bonds = self[0].get_all_bonds()
        return [first_image_bonds == image.get_all_bonds() for image in self]

    def find_holes(self):
        l = self.find_bonds_breaks()
        holes_list = []
        in_hole = False
        for i, value in enumerate(l):
            if value:
                in_hole = False
            else:
                if in_hole:
                    holes_list[-1][1] += 1
                else:
                    in_hole = True
                    holes_list.append([i, 1])
        return holes_list

    def fix_bonds_breaks(self):
        l = self.find_holes()

        for hole, length in l:
            # print(f'{hole = } {length = }')
            self[hole:hole+length] = self[hole -
                                          1].linear_interploation(self[hole+length], n=length).images

        l = self.find_holes()

        if len(l) > 0:
            warnings.warn(
                f'Linear interpolation didn\'t fix the problem. Holes:{l}')
            for hole, length in l:
                # print(f'{hole = } {length = }')
                for i in range(hole, hole+length):
                    p = self[hole-1].copy()
                    self[i] = p

        l = self.find_holes()

        if len(l) > 0:
            warnings.warn(f'Problem wasn\'t fixed. Holes:{l}')

    def to_xyz_string(self):
        return "".join([image.to_xyz_string() for image in self])

    def save(self, filename):
        with open(filename, "w") as text_file:
            text_file.write(self.to_xyz_string())

    def copy(self):
        return Path([image.copy() for image in self])

    def to_allxyz_string(self):
        return ">\n".join([image.to_xyz_string() for image in self])

    def save_as_allxyz(self, filename):
        with open(filename, "w") as text_file:
            text_file.write(self.to_allxyz_string())

    def __str__(self):
        info = [
            f"Path Information:",
            f"Number of Images: {len(self)}",
            f"Image Type: {self._get_type().__name__ if self._get_type() else 'None'}",
            f"First Image:\n{self[0] if len(self) > 0 else 'None'}",
            f"Last Image:\n{self[-1] if len(self) > 0 else 'None'}",
        ]

        return "\n".join(info)

    def render(self):
        current_idx = 0
        p = self[current_idx].render()
        print("Press 'n' to move to the next molecule, 'p' to go back, and 'q' to quit.")

        def key_press(obj, event):
            nonlocal current_idx
            key = obj.GetKeySym()  # Get the pressed key
            if key == 'n' and current_idx < len(self) - 1:
                current_idx += 1
            elif key == 'p' and current_idx > 0:
                current_idx -= 1
            elif key == 'q':
                # Exit the rendering
                p.close()
                return
            # Update the rendered molecule based on the current_idx
            p.clear()
            self[current_idx].render(p)
            p.reset_camera()
            p.render()

        p.iren.add_observer('KeyPressEvent', key_press)
        p.show()

    def render_alongside(self, other, alpha=1.0):
        current_idx = 0
        p = self[current_idx].render()

        if isinstance(other, Path):
            other[current_idx].render(p, alpha=alpha)
            print(
                "Press 'n' to move to the next pair of molecules, 'p' to go back, and 'q' to quit.")

            def key_press(obj, event):
                nonlocal current_idx
                key = obj.GetKeySym()  # Get the pressed key
                if key == 'n' and current_idx < min(len(self), len(other)) - 1:
                    current_idx += 1
                elif key == 'p' and current_idx > 0:
                    current_idx -= 1
                elif key == 'q':
                    # Exit the rendering
                    p.close()
                    return
                # Update the rendered molecules based on the current_idx
                p.clear()
                self[current_idx].render(p)
                other[current_idx].render(p, alpha=alpha)
                p.reset_camera()
                p.render()

            p.iren.add_observer('KeyPressEvent', key_press)
        elif isinstance(other, Atoms):
            other.render(p, alpha=alpha)
        else:
            raise TypeError(
                f"Expected other to be of type Path or ase.Atoms, got {type(other)}")

        p.show()
