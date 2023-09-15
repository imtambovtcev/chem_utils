from ase.collections import g2
from chem_utils.motor import Molecule, Fragment, Motor
import numpy as np


def test_molecule():
    Molecule(g2['C6H6'])


def test_motor():
    Motor(g2['C6H6'])


def test_fragment():
    m = Motor(g2['C6H6'])
    Fragment(m, attach_atom=0)


def test_reorder():
    m = Molecule(g2['C6H6'])
    new_m = m.get_standard_order()
    assert new_m.best_mapping(m) == {
        1: 1, 7: 0, 8: 2, 4: 7, 3: 5, 2: 3, 10: 6, 6: 8, 9: 4, 5: 11, 0: 9, 11: 10}


def test_rotation():
    m = Molecule(g2['C2H4'])
    new_m = m.rotate_part(0, [0, 1], 90)
    assert np.linalg.norm(new_m.get_positions()[0]-[0., 0., 0.66748]) < 1e5


def test_simple_bonds():
    m = Molecule(g2['C2H2'])
    m.update_bond_labels()
    assert m.G.edges[0, 1]['bond_type'] == 3


def test_ring_bonds():
    m = Molecule(g2['C6H6'])
    m.update_bond_labels()
    assert sum([m.G.edges[i, j]['bond_type']
               for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]]) == 9
