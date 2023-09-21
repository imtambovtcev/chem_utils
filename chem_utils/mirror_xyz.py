import sys
from pathlib import Path
from ase.io import read
from ase.io import write
import numpy as np
import math

from .format_xyz import format_xyz_file

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
