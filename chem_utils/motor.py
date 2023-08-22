from __future__ import annotations
import numpy as np
# from .find_cycles import find_bond
# from .rotate_path import rotation_matrix
from .render_molecule import render_molecule_from_atoms, render_molecule_from_path
from ase import Atoms
from ase.io import read, write
from ase.geometry.analysis import Analysis
import networkx as nx
import networkx.algorithms.isomorphism as iso
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from .format_xyz import format_xyz_file
# from .rotate_path import rotate_path
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import warnings
from sklearn.decomposition import PCA
import io


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


def simple_rotation(v1, v2):
    # v1 = self.fragment_attach_vector
    # v2 = vector
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    # print(f'{v1 = }')
    # print(f'{v2 = }')

    # Calculate the rotation vector and rotation angle
    rotation_vector = np.cross(v1, v2)
    rotation_angle = np.arccos(np.dot(v1, v2))
    rotation_angle = 0.0 if np.isnan(rotation_angle) else rotation_angle
    print(f'{rotation_angle = }')

    # Compute the rotation matrix
    return R.from_rotvec(rotation_angle * rotation_vector)


def best_rotation(v1, v2, v3, v4):
    # Ensure the vectors are numpy arrays and normalized
    v1 = np.array(v1) / np.linalg.norm(v1)
    v2 = np.array(v2) / np.linalg.norm(v2)
    v3 = np.array(v3) / np.linalg.norm(v3)
    v4 = np.array(v4) / np.linalg.norm(v4)

    # Calculate rotations
    rot1 = R.from_rotvec(np.cross(v3, v1))
    rot2 = R.from_rotvec(np.cross(v4, v2))

    # Apply first rotation to v4
    v4_rot = rot1.apply(v4)

    # Calculate the second rotation from rotated v4 to v2
    rot3 = R.from_rotvec(np.cross(v4_rot, v2))

    # Combine rotations
    rot_comb = rot3 * rot1

    # Return combined rotation
    return rot_comb


def best_fit_plane(points):
    # Вычисляем центроид (среднее всех точек)
    centroid = np.mean(points, axis=0)

    # Центрируем точки
    centered_points = points - centroid

    # Вычисляем ковариационную матрицу
    cov_matrix = np.cov(centered_points.T)

    # Применяем сингулярное разложение
    _, _, vh = np.linalg.svd(cov_matrix)

    # Нормаль к плоскости - это последний вектор в матрице vh
    normal = vh[-1]

    return centroid, normal


# def bonded_nodes_with_exclude(i,bonds, exclude=[]):
#     b = [bond[0] if bond[0]!=i else bond[1] for bond in bonds if i in bond]
#     return [bond for bon in b if bond not in exclude]


def first_split_node(i, G):
    # Check if the starting node 'i' itself is a split
    if len(list(G.neighbors(i))) > 1:
        return i, list(G.neighbors(i))

    visited = set()
    visited.add(i)

    # Use BFS to find the first split or the last node
    for node in nx.bfs_tree(G, i):
        visited.add(node)
        if len(list(G.neighbors(node))) > 2:
            next_level_nodes = set(G.neighbors(node)) - visited
            return node, list(next_level_nodes)

    return node, None


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


class Molecule:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms.copy()
        self.reinit()

    def reinit(self):
        self.ana = Analysis(self.atoms)
        self.symbols = set(self.atoms.get_chemical_symbols())
        self.bonds = [a for atom in self.symbols for x in self.symbols for a in self.ana.get_bonds(
            atom, x, unique=True)[0]]
        self.G = nx.Graph(self.bonds).to_directed()

    def copy(self):
        return Molecule(self.atoms.copy())

    def extend(self, atoms):
        self.atoms.extend(atoms)
        self.reinit()

    def get_coords(self, atoms=None):
        if atoms is None:
            return self.atoms.get_positions()
        else:
            return self.atoms.get_positions()[atoms, :]

    def set_coords(self, new_coords, atoms=None):
        if atoms is None:
            self.atoms.set_positions(new_coords)
        else:
            co = self.get_coords()
            co[atoms, :] = new_coords
            self.atoms.set_positions(co)

    def divide_in_two(self, bond):
        G = self.G.copy().to_undirected()
        G.remove_edge(bond[0], bond[1])
        return list(nx.shortest_path(G, bond[0]).keys()), list(nx.shortest_path(G, bond[1]).keys())

    def divide_in_two_fragments(self, fragment_attach_atom, fragment_atoms_ids):
        '''
        fragment_attach_atom - atom in fragment that is attached to the rest of the molecule
        fragment_atoms_ids - ids of the atoms fragments
        '''
        small_fragment_ids = list(set(fragment_atoms_ids))
        # print(f'{small_fragment_ids = }')
        main_fragment_ids = list(
            (set(range(len(self.atoms)))-set(small_fragment_ids)) | {fragment_attach_atom})
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
            new_small_fragment_ids[fragment_attach_atom], self.get_coords(small_fragment_ids))
        V0_m, V1_m, V2_m, V3_m = fragment_vectors(
            new_main_fragment_ids[fragment_attach_atom], self.get_coords(main_fragment_ids))
        # print(f'{self.atoms.symbols[small_fragment_ids] = }')

        R_s = np.column_stack([V1_s, V2_s, V3_s])
        R_m = np.column_stack([V1_m, V2_m, V3_m])
        # R_s = np.linalg.inv(np.column_stack([V1_s, V2_s, V3_s]))
        # R_m = np.linalg.inv(np.column_stack([V1_m, V2_m, V3_m]))
        # R_to_small = np.dot(R_s, np.linalg.inv(R_m))
        # R_to_main = np.linalg.inv(R_to_small)
        small_atoms = Atoms(
            self.atoms.symbols[small_fragment_ids], positions=self.get_coords(small_fragment_ids))
        main_atoms = Atoms(
            self.atoms.symbols[main_fragment_ids], positions=self.get_coords(main_fragment_ids))
        return Fragment(main_atoms, new_main_fragment_ids[fragment_attach_atom], R_s), Fragment(small_atoms, new_small_fragment_ids[fragment_attach_atom], R_m)

    def compare(self, other):
        return self.symbols

    def get_all_bonds(self):
        H = self.G.to_undirected()
        return list(H.edges())

    def get_bonded_atoms_of(self, i):
        H = self.G.to_undirected()
        return list(H[i])

    def get_bonds_of(self, i):
        if isinstance(i, int):
            return [bond for bond in self.bonds if i in self.get_all_bonds()]
        else:
            return [bond for bond in self.get_all_bonds() if len(set(bond) & set(i)) > 0]

    def id_to_symbol(self, l):
        c = self.atoms.get_chemical_symbols()
        try:
            return c[l]
        except:
            return [c[i] for i in l]

    def spatial_distance_between_atoms(self, i, j):
        if i is None or j is None:
            warnings.warn("Either 'i' or 'j' is None. Returning NaN.")
            return np.NaN
        c = self.atoms.get_positions()
        return np.linalg.norm(c[i, :]-c[j, :])

    def is_isomorphic(self, other):
        return nx.is_isomorphic(self.G, other.G)

    def all_isomorphisms(self, other):
        if not self.is_isomorphic(other):
            return None
        return nx.vf2pp_all_isomorphisms(self.G, other.G)

    def iso_distance(self, other, iso=None):
        coord_1 = self.atoms.get_positions()
        coord_2 = other.atoms.get_positions()
        if iso is not None:
            coord_1 = coord_1[np.array(list(iso.keys()), dtype=int), :]
            coord_2 = coord_2[np.array(list(iso.values()), dtype=int), :]
        # print(np.linalg.norm(cdist(coord_1, coord_1) - cdist(coord_2, coord_2)))
        return np.linalg.norm(cdist(coord_1, coord_1) - cdist(coord_2, coord_2))

    def best_isomorphism(self, other):
        if not (self.is_isomorphic(other)):
            return None
        return min(self.all_isomorphisms(other), key=lambda x: self.iso_distance(other, x))

    def __eq__(self, other):
        # print(f'{self.symbols == other.symbols = }')
        # print(f'{self.is_isomorphic(other) = }')
        if self.symbols != other.symbols:
            return False
        isos = self.all_isomorphisms(other)
        if isos is None:
            return False
        else:
            symb_1 = np.array(self.atoms.get_chemical_symbols())
            symb_2 = np.array(other.atoms.get_chemical_symbols())
            for iso in isos:
                # print(iso)
                # print(symb_1[np.array(list(iso.keys()), dtype=int)])
                # print(symb_2[np.array(list(iso.keys()), dtype=int)])
                if np.all(symb_1[np.array(list(iso.keys()), dtype=int)] == symb_2[np.array(list(iso.values()), dtype=int)]):
                    return True
            return False

    def best_distance(self, other):
        if self.is_isomorphic(other):
            new = other.reorder(self.best_isomorphism(other))
            return self.iso_distance(new)
        else:
            return np.nan

    def reorder(self, iso):
        order = np.arange(len(self.atoms))
        order[np.array(list(iso.keys()), dtype=int)
              ] = order[np.array(list(iso.values()), dtype=int)]
        coords = self.atoms.get_positions()[order, :]
        symbols = self.atoms.get_chemical_symbols()
        return Motor(Atoms(symbols=symbols, positions=coords))

    def get_bonds_without(self, elements=['H', 'F']):
        # exclude H and F
        symb = list(set(self.symbols)-set(elements))
        # bonds between non H and F
        return list(set([tuple(sorted(item)) for y in symb for x in symb for item in self.ana.get_bonds(
            y, x, unique=True)[0]]))

        # self.atoms = rotate_path(self.atoms)[0]

    def rotate_part(self, atom, bond, angle):
        # print(self.get_coords())
        edge = bond if bond[0] == atom else bond[::-1]
        print(edge)
        r0 = self.get_coords(atom)
        v = self.get_coords(edge[1])-r0
        island = self.divide_in_two(edge)[0]
        # print(island)
        r = rotation_matrix(v, np.pi*angle/180)
        new_coords = np.matmul(r, (self.get_coords(island)-r0).T).T+r0
        self.set_coords(new_coords, island)

        return self
        # print(self.get_coords())

    def render(self, show=True, show_numbers=False):
        p = render_molecule_from_atoms(self.atoms, show_numbers=show_numbers)
        if show:
            p.show()
        else:
            return p

    def render_alongside(self, other, alpha=1.0, show=True):
        p = render_molecule_from_atoms(self.atoms)
        p = render_molecule_from_atoms(other.atoms, p, alpha=alpha)
        # print(p)
        if show:
            p.show()
        else:
            return p

    def get_fragment(self, fragment_attach_atom, fragment_atoms_ids):
        '''
        Function to get fragment from a molecule
        fragment_attach_atom - atom in fragment that is attached to the rest of the molecule
        fragment_atoms_ids - ids of the atoms fragments
        '''
        _, fragment = self.divide_in_two_fragments(fragment_attach_atom, fragment_atoms_ids)
        return fragment

    # def replace_fragment(self, to_replace, replace_with, replacement_type='best_rotation'):
    #     # to be sure that the positioning is correct
    #     _to_replace = self.get_fragment(
    #         to_replace.fragment_attach_atom, to_replace.original_ids)
    #     new_molecule = self.copy()
    #     # print(f'{replace_with.atoms.positions = }')
    #     if replacement_type == 'simple_rotation':
    #         replace_with.move(_to_replace.attach_point -
    #                           replace_with.attach_point)
    #         rotation = simple_rotation(
    #             _to_replace.fragment_attach_vector, replace_with.fragment_attach_vector)  # check!
    #         replace_with.rotate_around_attach_point(rotation)
    #     elif replacement_type == 'best_rotation':
    #         replace_with.move(_to_replace.attach_point -
    #                           replace_with.attach_point)
    #         rotation = best_rotation(_to_replace.fragment_attach_vector, _to_replace.connection_normal,
    #                                  replace_with.fragment_attach_vector,  replace_with.connection_normal)
    #         replace_with.rotate_around_attach_point(rotation)
    #     # print(f'{replace_with.atoms.positions = }')
    #     # print(f'{_to_replace.original_ids = }')
    #     # print(f'{new_molecule.atoms = }')
    #     del new_molecule.atoms[_to_replace.original_ids]
    #     # print(f'{new_molecule.atoms = }')
    #     new_molecule.extend(replace_with.atoms)
    #     # print(f'{new_molecule.atoms = }')
    #     return new_molecule

    def rotate(self, zero_atom, x_atom, no_z_atom):
        '''
        rotate molecule by 3 atoms:
        - zero_atom: to be placed to (0,0,0)
        - x_atom: to be placed to (?, 0, 0)
        - no_z_atom: to be placed to (?, ?, 0)
        '''
        positions = self.atoms.get_positions()
        positions -= positions[zero_atom, :]
        positions = np.matmul(rotation_matrix(np.cross(positions[x_atom, :], [1, 0, 0]), np.arccos(
            np.dot(positions[x_atom, :], [1, 0, 0])/np.linalg.norm(positions[x_atom, :]))), positions.T).T
        positions = np.matmul(rotation_matrix(np.array(
            [1., 0., 0.]), -np.arctan2(positions[no_z_atom, 2], positions[no_z_atom, 1])), positions.T).T
        self.atoms.set_positions(positions)

    def atoms_to_xyz_string(self):
        sio = io.StringIO()
        write(sio, self.atoms, format="xyz")
        return sio.getvalue()


def read_fragment(filename):
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

        return Fragment(atoms, attach_atom, attach_matrix)


class Fragment(Molecule):
    def __init__(self, atoms: Atoms, attach_atom: int, attach_matrix=None):
        super().__init__(atoms)
        self.attach_atom = attach_atom
        if attach_matrix is None:
            self.attach_matrix = np.eye(3)
        else:
            self.attach_matrix = attach_matrix

    @property
    def attach_point(self):
        return self.atoms.positions[self.attach_atom]

    @property
    def fragment_vectors(self):
        return fragment_vectors(self.attach_atom, self.atoms.positions)

    def copy(self):
        return Fragment(self.atoms, self.attach_atom, self.attach_matrix)

    def _fragment_render(self, vectors=True, show_numbers=False):
        p = render_molecule_from_atoms(self.atoms, show_numbers=show_numbers)
        V0, V1, V2, V3 = self.fragment_vectors
        if vectors:
            add_vector_to_plotter(p, V0, V1, color='red')
            add_vector_to_plotter(p, V0, V2, color='green')
            add_vector_to_plotter(p, V0, V3, color='blue')
        return p

    def apply_transition(self, r0):
        self.atoms.positions = self.atoms.positions + r0

    def apply_rotation(self, R):
        # Rotate atom positions (assuming each row is a position vector)
        self.atoms.positions = np.dot(R, self.atoms.positions.T).T

        # Rotate the attach_matrix (assuming each column is a vector)
        self.attach_matrix = np.dot(R, self.attach_matrix)

    def get_origin_rotation_matrix(self):
        V0, V1, V2, V3 = self.fragment_vectors
        return np.linalg.inv(np.column_stack([V1, V2, V3]))

    def set_to_origin(self):
        self.apply_transition(-self.attach_point)
        self.apply_rotation(self.get_origin_rotation_matrix())

    def render(self, show=True, show_numbers=False):
        p = self._fragment_render(vectors=True, show_numbers=show_numbers)
        if show:
            p.show()
        else:
            return p

    def render_alongside(self, other, alpha=1.0, show=True):
        p = self._fragment_render(vectors=True)
        p = render_molecule_from_atoms(other.atoms, p, alpha=alpha)
        # print(p)
        if show:
            p.show()
        else:
            return p

    def save(self, filename):
        # Convert attach matrix to string
        matrix_string = ' '.join([' '.join(map(str, row))
                                 for row in self.attach_matrix])

        # Create the second line with embedded info
        fragment_info = f'Fragment: {self.attach_atom} {matrix_string}'

        # Construct the save string
        atom_info = self.atoms_to_xyz_string().split('\n', 1)[1].strip()
        save_string = self.atoms_to_xyz_string().split(
            '\n', 1)[0] + '\n' + fragment_info + '\n' + atom_info

        with open(filename, "w") as text_file:
            text_file.write(save_string)

    def connect_fragment(self, fragment: Fragment):
        _fragment = fragment.copy()
        _fragment.set_to_origin()
        _fragment.apply_rotation(self.attach_matrix)
        _fragment.apply_transition(self.attach_point)
        add_atoms = _fragment.atoms
        del add_atoms[_fragment.attach_atom]
        return Molecule(self.atoms+add_atoms)

    def __add__(self, fragment):
        return self.connect_fragment(fragment)


# Define the standard_stator graph
standard_stators = [[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 8), (4, 6), (4, 9), (5, 10), (6, 12), (8, 14), (9, 14), (10, 16), (12, 16)],
                    [(3, 7), (5, 7), (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 8), (4, 10),
                     (5, 11), (6, 12), (8, 14), (10, 14), (11, 18), (12, 18), (7, 3), (7, 5)],
                    [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 8), (4, 9), (4, 10), (5, 11), (6, 13), (6, 14), (8, 16), (9, 16), (10, 14), (11, 19), (13, 19)]]

standard_stators_list = []

for ss in standard_stators:
    standard_stator = nx.Graph()
    standard_stator.add_edges_from(ss)
    standard_stator = standard_stator.to_undirected()
    standard_stators_list.append(standard_stator)


class Motor(Molecule):
    def __init__(self, atoms):
        super().__init__(atoms)

    def find_bond(self):
        '''
        Identify the bond connecting the stator and rotor in a molecule.
        Returns:
        - dict: Information about the bond, stator, and rotor nodes and their neighbors.
        bond[0] -- stator
        bond[1] -- rotor
        '''

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
        # bonds = [item for x in self.symbols if x !='H' for item in self.ana.get_bonds('C', x, unique=True)[0]]+ [item for x in self.symbols if x !='H' for item in self.ana.get_bonds('S', x, unique=True)[0]]+ [item for x in self.symbols if x !='H' for item in self.ana.get_bonds('O', x, unique=True)[0]]
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

        if any(f_b['C_H_rotor'] in b for b in b_b):
            split_bond = [b for b in b_b if f_b['C_H_rotor'] in b]
            if not split_bond:
                return []  # Return empty list if split_bond is empty
            split_bond = split_bond[0]
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
        # Создаем неориентированные версии графов
        G_un = self.G.to_undirected()
        other_un = other.G.to_undirected()

        # Получаем номера замененных вершин
        old_nodes = [self.get_stator_rotor_bond()['bond_stator_node'],
                     self.get_stator_rotor_bond()['bond_rotor_node']]

        # Создаем подграф вокруг этих вершин в исходном графе
        old_subgraph_nodes = set(old_nodes)
        old_subgraph_edges = set()
        for node in old_nodes:
            adj_nodes = list(G_un.adj[node])
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
            adj_nodes = list(other_un.adj[node])
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
            ana = Analysis(self.atoms)
            bonds = self.get_bonds_without()
            # print(f'{bonds = }')
            stator_rotor = self.find_bond()
            zero_atom, x_atom, no_z_atom = stator_rotor['bond_stator_node'], stator_rotor[
                'stator_neighbours'][0], stator_rotor['stator_neighbours'][1]
        elif settings['mode'] == 'bond_x':
            ana = Analysis(self.atoms)
            bonds = self.get_bonds_without()
            # print(f'{bonds = }')
            stator_rotor = self.find_bond()
            zero_atom, x_atom, no_z_atom = stator_rotor['bond_stator_node'], stator_rotor[
                'bond_rotor_node'], np.min(stator_rotor['rotor_neighbours'])
        else:
            zero_atom, x_atom, no_z_atom = settings['zero_atom'], settings['x_atom'], settings['no_z_atom']

        return zero_atom, x_atom, no_z_atom
