import numpy as np
from .find_cycles import find_bond
from .rotate_path import rotation_matrix
from .render_molecule import render_molecule_from_path
from ase import Atoms
from ase.io import read
from ase.geometry.analysis import Analysis
import networkx as nx
import networkx.algorithms.isomorphism as iso
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from .format_xyz import format_xyz_file


class Motor:
    def __init__(self, atoms) -> None:
        self.atoms = atoms.copy()
        self.ana = Analysis(self.atoms)
        self.symbols = set(self.atoms.get_chemical_symbols())
        # print(self.symbols)
        self.bonds = self.get_all_bonds()
        self.G = nx.Graph(self.bonds).to_directed()

    def copy(self):
        return Motor(self.atoms.copy())

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

    def get_stator_rotor_bond(self):
        bonds = [item for x in self.symbols if x !=
                 'H' for item in self.ana.get_bonds('C', x, unique=True)[0]]
        return find_bond(bonds)

    def break_bonds(self):
        bonds = [item for x in self.symbols if x !=
                 'H' and x != 'F' for item in self.ana.get_bonds('C', x, unique=True)[0]]
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

    def break_nodes(self):
        rotor_node = self.get_stator_rotor_bond()['bond_rotor_node']
        return [b[np.argmax([nx.shortest_path_length(self.G, b[0], rotor_node), nx.shortest_path_length(self.G, b[1], rotor_node)])] for b in self.break_bonds()]

    def divide_in_two(self, bond):
        G = self.G.copy().to_undirected()
        G.remove_edge(bond[0], bond[1])
        return list(nx.shortest_path(G, bond[0]).keys()), list(nx.shortest_path(G, bond[1]).keys())

    def compare(self, other):
        return self.symbols

    def get_all_bonds(self):
        return [a for atom in self.symbols for x in self.symbols for a in self.ana.get_bonds(atom, x, unique=True)[0]]

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
        return sorted(self.all_isomorphisms(other), key=lambda x: self.iso_distance(other, x))[0]

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

    def save(self, filename):
        self.atoms.write(filename)
        format_xyz_file(filename)

    def render(self):
        p = render_molecule_from_path([self.atoms])
        print(p)
        p.show(window_size=[1000, 1000], cpos=cpos, jupyter_backend='panel')
