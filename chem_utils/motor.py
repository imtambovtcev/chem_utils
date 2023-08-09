import numpy as np
# from .find_cycles import find_bond
from .rotate_path import rotation_matrix
from .render_molecule import render_molecule_from_atoms, render_molecule_from_path
from ase import Atoms
from ase.io import read
from ase.geometry.analysis import Analysis
import networkx as nx
import networkx.algorithms.isomorphism as iso
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from .format_xyz import format_xyz_file
from .rotate_path import rotate_path
from scipy.spatial.transform import Rotation as R
import pyvista as pv


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
        return Motor(self.atoms.copy())

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

    def compare(self, other):
        return self.symbols

    def get_all_bonds(self):
        H = self.G.to_undirected()
        return list(H.edges())

    def get_bonds_of(self, i):
        H = self.G.to_undirected()
        return list(H[i])

    def id_to_symbol(self, l):
        c = self.atoms.get_chemical_symbols()
        try:
            return c[l]
        except:
            return [c[i] for i in l]

    def spatial_distance_between_atoms(self, i, j):
        if i is None or j is None:
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

    def rotate(self):
        self.atoms = rotate_path(self.atoms)[0]

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
        fragment_attach_vector_from_molecule = np.mean(self.get_coords(list(set(self.get_bonds_of(
            fragment_attach_atom))-set(fragment_atoms_ids)))-self.get_coords(fragment_attach_atom), axis=0)
        fragment_attach_vector_from_molecule /= - \
            np.linalg.norm(fragment_attach_vector_from_molecule)
        fragment_attach_vector_from_fragment = np.mean(self.get_coords(list(set(self.get_bonds_of(
            fragment_attach_atom)) and set(fragment_atoms_ids)))-self.get_coords(fragment_attach_atom), axis=0)
        fragment_attach_vector_from_fragment /= np.linalg.norm(
            fragment_attach_vector_from_fragment)
        # print(f'{fragment_attach_vector_from_molecule = }')
        # print(f'{fragment_attach_vector_from_fragment = }')
        # print(f'{np.arccos(np.dot(fragment_attach_vector_from_molecule, fragment_attach_vector_from_fragment)) = }')
        fragment_attach_vector = 0.5 * \
            (fragment_attach_vector_from_molecule +
             fragment_attach_vector_from_fragment)
        centroid, normal = best_fit_plane(self.get_coords(fragment_atoms_ids))
        connection_centroid, connection_normal = best_fit_plane(self.get_coords(list(set(
            self.get_bonds_of(fragment_attach_atom))-set(fragment_atoms_ids))+[fragment_attach_atom]))
        return Fragment(atoms=self.atoms[fragment_atoms_ids],
                        fragment_attach_atom=fragment_attach_atom,
                        fragment_attach_vector=fragment_attach_vector,
                        original_ids=fragment_atoms_ids,
                        centroid=centroid, normal=normal,
                        connection_centroid=connection_centroid, connection_normal=connection_normal)

    def replace_fragment(self, to_replace, replace_with, replacement_type='best_rotation'):
        # to be sure that the positioning is correct
        _to_replace = self.get_fragment(
            to_replace.fragment_attach_atom, to_replace.original_ids)
        new_molecule = self.copy()
        # print(f'{replace_with.atoms.positions = }')
        if replacement_type == 'simple_rotation':
            replace_with.move(_to_replace.attach_point -
                              replace_with.attach_point)
            rotation = simple_rotation(
                _to_replace.fragment_attach_vector, replace_with.fragment_attach_vector)  # check!
            replace_with.rotate_around_attach_point(rotation)
        elif replacement_type == 'best_rotation':
            replace_with.move(_to_replace.attach_point -
                              replace_with.attach_point)
            rotation = best_rotation(_to_replace.fragment_attach_vector, _to_replace.connection_normal,
                                     replace_with.fragment_attach_vector,  replace_with.connection_normal)
            replace_with.rotate_around_attach_point(rotation)
        # print(f'{replace_with.atoms.positions = }')
        # print(f'{_to_replace.original_ids = }')
        # print(f'{new_molecule.atoms = }')
        del new_molecule.atoms[_to_replace.original_ids]
        # print(f'{new_molecule.atoms = }')
        new_molecule.extend(replace_with.atoms)
        # print(f'{new_molecule.atoms = }')
        return new_molecule


class Fragment(Molecule):
    def __init__(self, atoms: Atoms, fragment_attach_atom: int, fragment_attach_vector: np.array, original_ids=None, centroid=None, normal=None, connection_centroid=None, connection_normal=None):
        super().__init__(atoms)
        self.fragment_attach_atom = fragment_attach_atom
        self.fragment_attach_vector = fragment_attach_vector
        self.original_ids = original_ids
        self.new_ids = dict(zip(original_ids, range(len(original_ids))))
        self.centroid = centroid
        self.normal = normal
        self.connection_centroid = connection_centroid
        self.connection_normal = connection_normal

    @property
    def attach_point(self):
        return self.atoms.positions[self.new_ids[self.fragment_attach_atom]]

    def move(self, r0):
        self.atoms.positions += r0
        self.centroid += r0
        self.connection_centroid += r0

    def rotate_around_attach_point(self, rotation):
        r0 = np.copy(self.attach_point)
        self.move(-r0)

        self.atoms.positions = rotation.apply(self.atoms.positions)
        self.fragment_attach_vector = rotation.apply(
            self.fragment_attach_vector)
        self.centroid = rotation.apply(self.centroid)
        self.normal = rotation.apply(self.normal)
        self.connection_centroid = rotation.apply(self.connection_centroid)
        self.connection_normal = rotation.apply(self.connection_normal)
        print(f'{r0 = }')
        self.move(r0)

    def move_and_rotate(self, r0, vector, rotation_type='best_rotation'):
        # print(f'{self.attach_point = }')
        # print(f'{r0 = }')
        self.atoms.positions -= self.attach_point

        if rotation_type == 'simple_rotation':
            v1 = self.fragment_attach_vector
            v2 = vector
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            # print(f'{v1 = }')
            # print(f'{v2 = }')

            # Calculate the rotation vector and rotation angle
            rotation_vector = np.cross(v1, v2)
            rotation_angle = np.arccos(np.dot(v1, v2))
            rotation_angle = 0.0 if np.isnan(
                rotation_angle) else rotation_angle
            print(f'{rotation_angle = }')

            # Compute the rotation matrix
            rotation = R.from_rotvec(rotation_angle * rotation_vector)
            self.atoms.positions = rotation.apply(self.atoms.positions)
            self.fragment_attach_vector = rotation.apply(
                self.fragment_attach_vector)
            self.atoms.positions += r0
        elif rotation_type == 'best_rotation':
            v1 = self.fragment_attach_vector
            v2 = vector
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            v3 /= np.linalg.norm(v3)
            v4 /= np.linalg.norm(v4)
            rotation = best_rotation(v1, v2, v3, v4)
            self.atoms.positions = rotation.apply(self.atoms.positions)
            self.fragment_attach_vector = rotation.apply(
                self.fragment_attach_vector)
            self.atoms.positions += r0

        # else:
        #     rotation = Rotation.identity()

        # Rotate the atoms

    def _fragment_render(self, vectors=True, show_numbers=False):
        p = render_molecule_from_atoms(self.atoms, show_numbers=show_numbers)
        if vectors:
            add_vector_to_plotter(p, self.attach_point,
                                  self.fragment_attach_vector)
            add_vector_to_plotter(p, self.centroid, self.normal, color='red')
            add_vector_to_plotter(
                p, self.connection_centroid, self.connection_normal, color='green')
        return p

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

        edges = [item for x in self.symbols if x != 'H' for item in self.ana.get_bonds('C', x, unique=True)[0]] + [item for x in self.symbols if x != 'H' for item in self.ana.get_bonds(
            'S', x, unique=True)[0]] + [item for x in self.symbols if x != 'H' for item in self.ana.get_bonds('O', x, unique=True)[0]]
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
        if len(set(('H', 'F')) & set(self.id_to_symbol(self.get_bonds_of(f_b['rotor_neighbours'][0])))) > 0:
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
        # for i in self.get_bonds_of(f_b['C_H_stator']):
        #     print(i)
        #     if len(set(('H','F')) & set(self.id_to_symbol([i])))>0:
        #         print('Yes')
        return [i for i in self.get_bonds_of(f_b['C_H_rotor']) if len(set(('H', 'F')) & set(self.id_to_symbol([i]))) > 0][0]

    def get_stator_N(self):
        f_b = self.get_stator_rotor_bond()
        H = self.G.copy()
        H.remove_node(f_b['C_H_stator'])
        l = self.get_bonds_of(f_b['C_H_stator'])
        l.remove(f_b['bond_stator_node'])
        if nx.shortest_path_length(H, l[0], f_b['bond_stator_node']) > nx.shortest_path_length(H, l[1], f_b['bond_stator_node']):
            return l[0]
        else:
            return l[1]

    def get_stator_H(self):
        l = [i for i in self.get_bonds_of(self.get_stator_N()) if len(
            set(('H', 'F')) & set(self.id_to_symbol([i]))) > 0]
        return l[0] if len(l) > 0 else None

    def get_break_bonds(self):
        # exclude H and F
        symb = list(set(self.symbols)-{'H', 'F'})
        # bonds between non H and F
        bonds = list(set([tuple(sorted(item)) for y in symb for x in symb for item in self.ana.get_bonds(
            y, x, unique=True)[0]]))
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
            split_bond = [b for b in b_b if f_b['C_H_rotor'] in b][0]
        else: #len(b_b) == 0
            for i in self.G[f_b['C_H_rotor']]:
                # print(f'{i = }')
                tail_C = i
                # print(list(self.G[tail_C]))
                tail_H = [j for j in list(self.G[tail_C]) if self.id_to_symbol(
                    j) == 'H' or self.id_to_symbol(j) == 'F']
                # print(f'{tail_H = }')
                if len(tail_H) == 3:
                    return tail_H
            tail_start = list(split_bond)
            tail_start.remove(f_b['C_H_rotor'])
            tail_start = tail_start[0]
            sp = self.divide_in_two((f_b['C_H_rotor'], tail_start))
            H = self.G.copy()
            for i in sp[0]:
                H.remove_node(i)
            tail_distance = nx.shortest_path_length(H, source=tail_start)
            max_distance = np.max(list(tail_distance.values()))

            # print(max_distance)
            return [i for i in tail_distance.keys() if tail_distance[i] == max_distance]
            # tail= sp[1] if f_b['C_H_rotor'] in sp[0] else sp[0]

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
