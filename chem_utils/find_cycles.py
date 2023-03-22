import networkx as nx
import numpy as np


# def mutial_edge_and_neighbours(G, cycles):
#     bond = [[i, j] for j in cycles[1]
#             for i in cycles[0] if G.has_edge(i, j)][0]
#     cycle_neighbours = [j for _, j in G.out_edges(
#         [bond[0]]) if j is not bond[1]]
#     return bond[0], bond[1], np.min(cycle_neighbours)


# def find_connected_cycles(edges, l=5):
#     '''
#         returns info on 2 cycles if len l, connected by edge
#         first output node with edge in cycle 1
#         second output node with edge in cycle 2
#         third ourput is heighbour of output 1 with lower number
#     '''
#     G = nx.Graph(edges).to_directed()
#     cycles = sorted(nx.simple_cycles(G))
#     cycles = [sorted(i) for i in cycles if len(i) == l]
#     cycles = [list(x) for x in set(tuple(x) for x in cycles)]

#     # print(cycles)

#     return mutial_edge_and_neighbours(G, cycles)


# # def mutial_edge_and_neighbours(G, cycles):

def find_bond(edges):
    '''
        returns info on 2 parts of molecule connected by bond. Assumed that stator is fluorene-ish
    '''
    G = nx.Graph(edges).to_directed(
    )  # directed graph of edges (to search for cycles)
    cycles = sorted(nx.simple_cycles(G), key=lambda x: len(x), reverse=True)
    # print(cycles[0])

    # cycles not including biggest one, sorted as initial cycle was sorted
    small_cyclce = [c for c in cycles if not bool(set(c) & set(cycles[0]))]

    # print(small_cyclce[0])
    for a in small_cyclce[0]:
        b = [j for _, j in G.out_edges(a) if j in cycles[0]]
        if len(b) > 0:
            # bond between stator and rotor, initially assumed 0 is stator, 1 is rotor
            bond = [b[0], a]
            break

    # print(f'{bond = }')

    flat_cycles_6 = set([item for i in cycles if len(
        i) == 6 for item in i])  # atoms in 6-rings

    stator_neighbours = sorted(
        [j for _, j in G.out_edges(bond[0]) if j != bond[1]])
    rotor_neighbours = sorted(
        [j for _, j in G.out_edges(bond[1]) if j != bond[0]])

    # check for false (both neighbours in 6 ring = stator)
    if not (stator_neighbours[0] in flat_cycles_6 and stator_neighbours[1] in flat_cycles_6):
        stator_neighbours, rotor_neighbours = rotor_neighbours, stator_neighbours
        bond = bond[::-1]

    return {'bond': bond, 'bond_stator_node': bond[0], 'bond_rotor_node': bond[1], 'stator_neighbours': stator_neighbours, 'rotor_neighbours': rotor_neighbours}


def find_stator(edges):
    '''
        returns info on 2 cycles if len 5, connected by edge
        first output node with edge in cycle, connected to 2 6-rings (stator)
        second output node with edge in second cycle (rotor)
        third ourput is heighbour of output 1 with lower number
    '''
    G = nx.Graph(edges).to_directed()
    cycles = sorted(nx.simple_cycles(G))

    cycles_5 = [sorted(i) for i in cycles if len(i) == 5]
    cycles_5 = [list(x) for x in set(tuple(x) for x in cycles_5)]
    # print(f'{len(cycles_5) = }')
    if len(cycles_5) == 2:
        flat_cycles_6 = set([item for i in cycles if len(
            i) == 6 for item in i])  # atoms in 6-rings
        # print(cycles_5)
        # print(flat_cycles_6)

        bond = [[i, j] for j in cycles_5[1]
                for i in cycles_5[0] if G.has_edge(i, j)][0]

        # print(f'{bond = }')

        stator_neighbours = [
            j for _, j in G.out_edges(bond[0]) if j != bond[1]]
        # print(f'{G.out_edges(bond[0]) = }')
        # print(f'{bond[1]}')
        # print(f'{stator_neighbours = }')
        # print(f'{stator_neighbours[0] in flat_cycles_6 = }')
        # print(f'{stator_neighbours[1] in flat_cycles_6 = }')
        # both neighbours in 6 ring = stator
        if stator_neighbours[0] in flat_cycles_6 and stator_neighbours[1] in flat_cycles_6:
            # print('0 is stator')
            stator_node = bond[0]
            rotor_node = bond[1]
        else:
            # print('1 is stator')
            stator_node = bond[1]
            rotor_node = bond[0]
            stator_neighbours = [
                j for _, j in G.out_edges(bond[1]) if j != bond[0]]
        # print(f'{stator_neighbours = }')
        # print(stator_node, np.min(stator_neighbours), np.max(stator_neighbours))
        return stator_node, np.min(stator_neighbours), np.max(stator_neighbours)
    else:
        return cycles_5[0][1], cycles_5[0][0], cycles_5[0][2]

    # cycles = [sorted(i) for i in cycles if len(i) == 5]
    # cycles = [list(x) for x in set(tuple(x) for x in cycles)]

    # bond = [[i, j] for j in cycles[1]
    #         for i in cycles[0] if G.has_edge(i, j)][0]
    # cycle_neighbours = [j for _, j in G.out_edges(
    #     [bond[0]]) if j is not bond[1]]
    # return bond[0], bond[1], np.min(cycle_neighbours)


# edges = [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (3, 26), (4, 5), (4, 19), (5, 6), (6, 7), (6, 18), (7, 8), (7, 12), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 18), (14, 15), (15, 16), (16, 17), (17, 18), (19, 20), (19, 24), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26)]


# print(find_stator(edges))
