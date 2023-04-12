import sys
import pathlib
from pathlib import Path
import ase
from ase.io import read, write
from ase.geometry.analysis import Analysis

import numpy as np
import math
from .find_cycles import find_bond


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


def rotate_path(p, settings=None):
    # print(isinstance(p, list))
    if isinstance(p, list):
        path = [atoms.copy() for atoms in p]
        mode = 'return'
        # print('1')
    elif isinstance(p, ase.Atoms):
        path = [p.copy()]
        mode = 'return'
        # print('2')
    else:
        path = read(str(p), index=':')
        mode = 'save'
        # print('3')

    # print(f'{len(path) = }')
    # print(path)

    if settings is None or settings['mode'] == 'default':
        ana = Analysis(path[0])
        bonds = [item for x in list(set(path[0].get_chemical_symbols(
        ))) if x != 'H' for item in ana.get_bonds('C', x, unique=True)[0]]
        # print(f'{bonds = }')
        stator_rotor = find_bond(bonds)
        zero_atom, x_atom, no_z_atom = stator_rotor['bond_stator_node'], stator_rotor[
            'stator_neighbours'][0], stator_rotor['stator_neighbours'][1]
    elif settings['mode'] == 'bond_x':
        ana = Analysis(path[0])
        bonds = [item for x in list(set(path[0].get_chemical_symbols(
        ))) if x != 'H' for item in ana.get_bonds('C', x, unique=True)[0]]
        # print(f'{bonds = }')
        stator_rotor = find_bond(bonds)
        zero_atom, x_atom, no_z_atom = stator_rotor['bond_stator_node'], stator_rotor[
            'bond_rotor_node'], np.min(stator_rotor['rotor_neighbours'])
    else:
        zero_atom, x_atom, no_z_atom = settings['zero_atom'], settings['x_atom'], settings['no_z_atom']

    for atoms in path:
        positions = atoms.get_positions()
        positions -= positions[zero_atom, :]
        # print(positions.shape)
        # print(positions[2,:])
        # print(np.dot(positions[2,:],[1,0,0])/np.linalg.norm(positions[2,:]))
        positions = np.matmul(rotation_matrix(np.cross(positions[x_atom, :], [1, 0, 0]), np.arccos(
            np.dot(positions[x_atom, :], [1, 0, 0])/np.linalg.norm(positions[x_atom, :]))), positions.T).T
        positions = np.matmul(rotation_matrix(np.array(
            [1., 0., 0.]), -np.arctan2(positions[no_z_atom, 2], positions[no_z_atom, 1])), positions.T).T
        # print(f'{positions[zero_atom, :] = }')
        # print(f'{positions[x_atom, :] = }')
        # print(f'{positions[no_z_atom, :] = }')
        atoms.set_positions(positions)

    if mode == 'return':
        return path
    write(str(p)[:-4]+'_rotated.xyz', path)


def main():
    _input = ['./'] if len(sys.argv) <= 1 else sys.argv[1:]
    print(_input)
    _input = [Path(d) for d in _input]
    input = []
    for d in _input:
        if d.is_dir():
            add = d.glob('*.xyz')
            input.extend(add)
        else:
            input.append(d)
    print(input)
    [rotate_path(p) for p in input]


if __name__ == "__main__":
    main()
