from __future__ import annotations

import networkx as nx
import numpy as np

from .molecule import Molecule

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            print(f'{stator_neighbours=}')
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

    def get_stator_rotor_bond_vector(self):
        f_b = self.get_stator_rotor_bond()
        position_i = self.positions[f_b['bond_stator_node']]
        position_j = self.positions[f_b['bond_rotor_node']]
        return position_j - position_i

    def get_stator_rotor_ids(self):
        f_b = self.get_stator_rotor_bond()
        return self.divide_in_two(f_b['bond'])

    def get_stator_rotor(self, add_bond_to=None):
        f_b = self.get_stator_rotor_bond()
        stator, rotor = self.divide_in_two(f_b['bond'])
        stator = self.get_fragment(f_b['bond_stator_node'], stator)
        rotor = self.get_fragment(f_b['bond_rotor_node'], rotor)
        return stator, rotor

    def get_stator_rotor_with_bond(self):
        f_b = self.get_stator_rotor_bond()
        stator_list, _ = self.divide_in_two(f_b['bond'])
        rotor, stator = self.divide_in_two_fragments(
            f_b['bond_stator_node'], stator_list)
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

    def to_origin(self, zero_atom=None, x_atom=None, no_z_atom=None):
        if all(variable is None for variable in (zero_atom, x_atom, no_z_atom)):
            zero_atom, x_atom, no_z_atom = self.find_rotation_atoms()
        return super().to_origin(zero_atom, x_atom, no_z_atom)

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
