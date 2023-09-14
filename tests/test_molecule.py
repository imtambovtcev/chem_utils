from ase.collections import g2
from chem_utils.motor import Molecule, Fragment, Motor


def test_molecule():
    Molecule(g2['C6H6'])


def test_motor():
    Motor(g2['C6H6'])


def test_fragment():
    m = Motor(g2['C6H6'])
    Fragment(m, attach_atom=0)


def test_simple_bonds():
    m = Molecule(g2['C2H2'])
    m.update_bond_labels()
    assert m.G.edges[0,1]['bond_type'] == 3


def test_ring_bonds():
    m = Molecule(g2['C6H6'])
    m.update_bond_labels()
    assert sum([m.G.edges[i,j]['bond_type'] for i,j in [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]]) == 9
