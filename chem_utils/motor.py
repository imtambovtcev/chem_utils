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
        self.bonds = [a for atom in self.symbols for x in self.symbols for a in self.ana.get_bonds(
            atom, x, unique=True)[0]]
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
        f_b=find_bond(bonds)
        if len(set(('H','F')) & set(self.id_to_symbol(self.get_bonds_of(f_b['rotor_neighbours'][0]))))>0: #'H' or 'F' bonded to this atom
            f_b['C_H_rotor']=f_b['rotor_neighbours'][0]
            f_b['not_C_H_rotor']=f_b['rotor_neighbours'][1]
        else:
            f_b['C_H_rotor']=f_b['rotor_neighbours'][1]
            f_b['not_C_H_rotor']=f_b['rotor_neighbours'][0]
        if self.spatial_distance_between_atoms(f_b['C_H_rotor'], f_b['stator_neighbours'][0]) < self.spatial_distance_between_atoms(f_b['C_H_rotor'], f_b['stator_neighbours'][1]):
             f_b['C_H_stator']=f_b['stator_neighbours'][0]
             f_b['not_C_H_stator']=f_b['stator_neighbours'][1]
        else:
             f_b['C_H_stator']=f_b['stator_neighbours'][1]
             f_b['not_C_H_stator']=f_b['stator_neighbours'][0]
        return f_b

    def get_rotor_H(self):
        f_b=self.get_stator_rotor_bond()
        # for i in self.get_bonds_of(f_b['C_H_stator']):
        #     print(i)
        #     if len(set(('H','F')) & set(self.id_to_symbol([i])))>0:
        #         print('Yes')
        return [i for i in self.get_bonds_of(f_b['C_H_rotor']) if len(set(('H','F')) & set(self.id_to_symbol([i])))>0][0]

    def get_stator_N(self):
        f_b=self.get_stator_rotor_bond()
        H = self.G.copy()
        H.remove_node(f_b['C_H_stator'])
        l=self.get_bonds_of(f_b['C_H_stator'])
        l.remove(f_b['bond_stator_node'])
        if nx.shortest_path_length(H,l[0],f_b['bond_stator_node'])>nx.shortest_path_length(H,l[1],f_b['bond_stator_node']):
            return l[0]
        else:
            return l[1]

    def get_stator_H(self):
        l = [i for i in self.get_bonds_of(self.get_stator_N()) if len(set(('H','F')) & set(self.id_to_symbol([i])))>0]
        return l[0] if len(l)>0 else None

    def get_break_bonds(self):
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

    def get_break_nodes(self):
        rotor_node = self.get_stator_rotor_bond()['bond_rotor_node']
        return [b[np.argmax([nx.shortest_path_length(self.G, b[0], rotor_node), nx.shortest_path_length(self.G, b[1], rotor_node)])] for b in self.get_break_bonds()]

    def divide_in_two(self, bond):
        G = self.G.copy().to_undirected()
        G.remove_edge(bond[0], bond[1])
        return list(nx.shortest_path(G, bond[0]).keys()), list(nx.shortest_path(G, bond[1]).keys())

    def get_tails(self):
        f_b=self.get_stator_rotor_bond()
        b_b=self.get_break_bonds()
        if len(b_b)==0:
            for i in self.G[f_b['C_H_rotor']]:
                # print(f'{i = }')
                tail_C = i
                # print(list(self.G[tail_C]))
                tail_H = [j for j in list(self.G[tail_C]) if self.id_to_symbol(j)=='H' or self.id_to_symbol(j)=='F']
                # print(f'{tail_H = }')
                if len(tail_H) == 3:
                    return tail_H
        else:
            split_bond=[b for b in b_b if f_b['C_H_rotor'] in b][0]
            tail_start=list(split_bond)
            tail_start.remove(f_b['C_H_rotor'])
            tail_start=tail_start[0]
            sp=self.divide_in_two((f_b['C_H_rotor'],tail_start))
            H=self.G.copy()
            for i in sp[0]: H.remove_node(i)
            tail_distance=nx.shortest_path_length(H, source=tail_start)
            max_distance=np.max(list(tail_distance.values()))

            # print(max_distance)
            return [i for i in tail_distance.keys() if tail_distance[i]==max_distance]
            # tail= sp[1] if f_b['C_H_rotor'] in sp[0] else sp[0]


    def compare(self, other):
        return self.symbols

    def get_all_bonds(self):
        H = self.G.to_undirected()
        return list(H.edges())

    def get_bonds_of(self, i):
        H = self.G.to_undirected()
        return list(H[i])

    def id_to_symbol(self, l):
        c=self.atoms.get_chemical_symbols()
        try:
            return c[l]
        except:
            return [c[i] for i in l]

    def spatial_distance_between_atoms(self, i,j):
        if i is None or j is None:
            return np.NaN
        c=self.atoms.get_positions()
        return np.linalg.norm(c[i,:]-c[j,:])

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

    def render(self):
        p = render_molecule_from_path([self.atoms])
        print(p)
        p.show(window_size=[1000, 1000], cpos=cpos, jupyter_backend='panel')

    def render_alongside(self, other):
        p = render_molecule_from_path([self.atoms])
        p = render_molecule_from_path([other.atoms], p)
        print(p)
        p.show(window_size=[1000, 1000], cpos=cpos, jupyter_backend='panel')
