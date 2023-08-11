import sys
import pathlib
from pathlib import Path
import ase
from ase.io import read, write
from ase.geometry.analysis import Analysis
from .motor import Motor

import numpy as np
import math
# from .find_cycles import find_bond


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
    mode = set()

    if isinstance(p, list):
        path = [Motor(atoms) for atoms in p]
        mode.add('return')
        # print('1')
    elif isinstance(p, ase.Atoms):
        path = [Motor(p)]
        mode.add('return')
        # print('2')
    else:
        path = read(str(p), index=':')
        path = [Motor(atoms) for atoms in path]
        mode.add('save')
        # print('3')

    zero_atom, x_atom, no_z_atom = path[0].find_rotation_atoms(settings=settings)
    [motor.rotate(zero_atom, x_atom, no_z_atom) for motor in path]


    return_path = [motor.atoms for motor in path]

    if 'return' in mode:
        return return_path
    if 'save' in mode:
        write(str(p)[:-4]+'_rotated.xyz', return_path)


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
