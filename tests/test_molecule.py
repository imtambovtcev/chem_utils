from ase.collections import g2
from chem_utils import *
import numpy as np
import os


def test_molecule():
    Molecule(g2['C6H6'])


def test_motor():
    Motor(g2['C6H6'])


def test_fragment():
    m = Motor(g2['C6H6'])
    Fragment(m, attach_atom=0)


def test_path():
    m1 = Motor(g2['C6H6'])
    m2 = Motor(g2['C6H6'])
    Path([m1, m2])


def test_reorder():
    m = Molecule(g2['C6H6'])
    new_m = m.get_standard_order()
    assert new_m.best_mapping(m) == {
        1: 1, 7: 0, 8: 2, 4: 7, 3: 5, 2: 3, 10: 6, 6: 8, 9: 4, 5: 11, 0: 9, 11: 10}


def test_render():
    m = Molecule(g2['C6H6'])
    m.render()


def test_eldens():
    ElectronDensity.load('tests/C2H4.eldens.cube')


def test_eldens_render():
    m = ElectronDensity.load('tests/C2H4.eldens.cube')
    m.render()


def test_molecule_from_eldens():
    m = Molecule.load_from_cube('tests/C2H4.eldens.cube')
    m.rotate(np.array([
        [np.cos(np.radians(45)), -np.sin(np.radians(45)), 0],
        [np.sin(np.radians(45)), np.cos(np.radians(45)), 0],
        [0, 0, 1]]))
    m.translate(np.array([1., 0., 0.]))
    assert np.allclose(m.positions, np.array([[0.12247267, -0.59563596, -3.17528129],
                                             [-0.0358741,  0.14036158, -2.11545416],
                                             [0.16072858, - 1.67115906, -3.08479992],
                                             [0.24072292, - 0.13057927, -4.14282414],
                                             [-0.63260717, -0.22125846, -1.29111143],
                                              [0.44973282,  1.10274504, -2.04909693]]))


def test_molecule_from_eldens_render():
    m = Molecule.load_from_cube('tests/C2H4.eldens.cube')
    m.render()


def test_molecule_save_load():
    os.remove('tests/C6H6.xyz')
    m = Molecule(g2['C6H6'])
    m.save('tests/C6H6.xyz')
    m = Molecule.load('tests/C6H6.xyz')
    assert m.get_chemical_formula() == 'C6H6'


def test_rotation():
    m = Molecule(g2['C2H4'])
    new_m = m.rotate_part(0, [0, 1], 90)
    assert np.allclose(new_m.get_positions()[0],[0., 0., 0.66748])


def test_simple_bonds():
    m = Molecule(g2['C2H2'])
    m.update_bond_labels()
    assert m.G.edges[0, 1]['bond_type'] == 3


def test_ring_bonds():
    m = Molecule(g2['C6H6'])
    m.update_bond_labels()
    assert sum([m.G.edges[i, j]['bond_type']
               for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]]) == 9
