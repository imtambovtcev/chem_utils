import sys
from pathlib import Path
from ase.io import read
from ase.io import write
import numpy as np
import math

from .format_xyz import format_xyz_file


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


def mirror_pathlist(pathlist: list):
    for p in pathlist:
        path = read(p, index=':')

        print(path)

        for atoms in path:
            positions = atoms.get_positions()
            positions[:, 0] = -positions[:, 0]
            print(positions.shape)
            atoms.set_positions(positions)

        write(p.parent / (p.stem + '_mirrored.xyz'), path)
        format_xyz_file(p.parent / (p.stem + '_mirrored.xyz'))

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
    mirror_pathlist(input)



if __name__ == "__main__":
    main()
