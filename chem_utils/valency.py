import networkx as nx
import random

TYPICAL_VALENCIES = {
    'H': 1,    # Hydrogen
    'C': 4,    # Carbon
    'N': 3,    # Nitrogen
    'O': 2,    # Oxygen
    'F': 1,    # Fluorine
    'Cl': 1,   # Chlorine
    'Br': 1,   # Bromine
    'I': 1,    # Iodine
    'S': 2,    # Sulfur (can also be 4 or 6 in some compounds)
    'P': 3,    # Phosphorus (can also be 5 in some compounds)
    'Si': 4,   # Silicon
    'B': 3,    # Boron
    'Li': 1,   # Lithium
    'Na': 1,   # Sodium
    'K': 1,    # Potassium
    'Mg': 2,   # Magnesium
    'Ca': 2,   # Calcium
    'Fe': 2,   # Iron (can also be 3 in some compounds)
    'Cu': 2,   # Copper (can also be 1 or 3 in some compounds)
    'Zn': 2,   # Zinc
    'Al': 3,   # Aluminum
    'Ag': 1,   # Silver
    'Au': 1,   # Gold
    'Pt': 2,   # Platinum
    'Hg': 2,   # Mercury
    'Sn': 2,   # Tin (can also be 4 in some compounds)
    'Pb': 2,   # Lead (can also be 4 in some compounds)
    'Bi': 3,   # Bismuth
    'As': 3,   # Arsenic (can also be 5 in some compounds)
    'Se': 2,   # Selenium
    'Co': 2,   # Cobalt (can also be 3 in some compounds)
    'Ni': 2,   # Nickel
    'Mn': 2,   # Manganese (can also be 4 or 7 in some compounds)
    'Cr': 2,   # Chromium (can also be 3 or 6 in some compounds)
    'Mo': 4,   # Molybdenum
    'W': 6,    # Tungsten
    'V': 5,    # Vanadium (can also be 2, 3, or 4 in some compounds)
    # Add other elements as needed.
}


def split_integer(i, n):
    # Ensure that the generated numbers sum to i - n to account for adding 1 to each partition.
    random_numbers = [0] + sorted([random.randint(0, i - n) for _ in range(n - 1)]) + [i - n]
    return [random_numbers[j+1] - random_numbers[j] + 1 for j in range(n)]

def general_print(graph):
    output_strings = []
    for i, j in graph.edges():
        atom1_symbol = graph.nodes[i]['label']
        atom1_valency = graph.nodes[i].get('valency', '?')
        atom2_symbol = graph.nodes[j]['label']
        atom2_valency = graph.nodes[j].get('valency', '?')
        # Fetching bond label; if not present, use an empty string
        bond_label = graph[i][j].get('bond_type', '')
        # If bond_label is numeric, keep it as-is, otherwise use the string representation
        output_strings.append(
            f"{i} - {atom1_symbol}[{atom1_valency}] ({bond_label}) {atom2_symbol}[{atom2_valency}] - {j}")
    return "\n".join(output_strings)


def add_valency(H):
    for i in H.nodes():
            H.nodes[i]['valency'] = TYPICAL_VALENCIES[H.nodes[i]['label']]
    return H

def get_connected_free_atoms(H, i):
    return [j for j in H.neighbors(i) if H.nodes[j]['valency'] > 0]


def adjust_labels_in_H_based_on_equality(H):
    run_process = True
    changes_made = False
    # Initialize all edges with bond_type of 0

    while run_process:
        run_process = False

        for i in H.nodes():
            if H.nodes[i]['valency'] != 0:
                # print(f'{i = }')

                # number on the atom == number of bonds
                # print(f'{get_connected_free_atoms(H,i) = }')
                if H.nodes[i]['valency'] == len(get_connected_free_atoms(H, i)):
                    H.nodes[i]['valency'] = 0
                    for j in get_connected_free_atoms(H, i):
                        # print(f'{j = }')
                        H.nodes[j]['valency'] -= 1
                        assert H.nodes[j]['valency'] >= 0
                        H.edges[i, j]['bond_type'] += 1
                        run_process = True
                        changes_made = True
    return H, changes_made


def adjust_labels_in_H_based_on_single_bond(H):
    run_process = True
    changes_made = False
    # Initialize all edges with bond_type of 0

    while run_process:
        run_process = False

        for i in H.nodes():
            if H.nodes[i]['valency'] != 0:
                # print(f'{i = }')
                # print(f'{get_connected_free_atoms(H,i) = }')
                if len(get_connected_free_atoms(H, i)) == 1:
                    j = get_connected_free_atoms(H, i)[0]
                    # print(f'{j = }')
                    H.edges[i, j]['bond_type'] += H.nodes[i]['valency']
                    H.nodes[j]['valency'] -= H.nodes[i]['valency']
                    H.nodes[i]['valency'] = 0
                    run_process = True
                    changes_made = True

    return H, changes_made


def adjust_labels_in_H(H):
    changes_made = False
    run_process = True
    while run_process:
        run_process = False
        H, new_changes = adjust_labels_in_H_based_on_equality(H)
        # print(f'adjust_labels_in_H_based_on_equality {new_changes = }')
        run_process = run_process or new_changes
        # print("Graph H:\n", general_print(H))
        H, new_changes = adjust_labels_in_H_based_on_single_bond(H)
        run_process = run_process or new_changes
        # print(f'adjust_labels_in_H_based_on_single_bond {new_changes = }')
        # print("Graph H:\n", general_print(H))
        changes_made = run_process or changes_made
    return H, changes_made

def randomly_assign_bonds_of_one_atom(H):
    atoms = [i for i in H.nodes() if H.nodes[i]['valency'] != 0]
    atom = random.choice(atoms)
    neigbors = get_connected_free_atoms(H, atom)
    for j,n in zip(neigbors, split_integer(H.nodes[atom]['valency'], len(neigbors))):
        H.edges[atom, j]['bond_type'] += n
        H.nodes[j]['valency'] -= n
        H.nodes[atom]['valency'] -= n

    return H


def all_bonds_assigned(H):
    return not(any(data.get('bond_type') == 0 or data.get('bond_type') is None for _, _, data in H.edges(data=True)))

def rebond(G, random_attempts_n=100):
    H = G.copy()
    for i in H.nodes():
            H.nodes[i]['valency'] = TYPICAL_VALENCIES[H.nodes[i]['label']]
        
    for i, j in H.edges():
        H.edges[i, j]['bond_type'] = 0

    # print("Graph H:\n", general_print(H))

    random_copy=None
    random_attempts_counter=0
    while not(all_bonds_assigned(H)) and (random_attempts_n is None or random_attempts_counter<random_attempts_n):
        H, changes_made = adjust_labels_in_H(H)

        if not(changes_made):
            # print('random iteration')
            if random_copy is None:
                random_copy = H.copy()
            try:
                H = randomly_assign_bonds_of_one_atom(H)
            except Exception as e:
                print(e)
                H = random_copy
                random_attempts_counter+=1
        # print(f'{changes_made = }')

    return H
