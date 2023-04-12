import sys
from pathlib import Path
import ase
from ase.io import read, write

from .format_xyz import format_xyz_file


def get_image(p, save, n=0):
    # print(isinstance(p, list))
    if isinstance(p, list):
        path = [atoms.copy() for atoms in p]
    elif isinstance(p, ase.Atoms):
        path = [p.copy()]
    else:
        path = read(str(p), index=':')

    atoms = path[n]

    write(save, atoms)
    format_xyz_file(save)


def main():
    p = sys.argv[-3]
    save = sys.argv[-2]
    n = int(sys.argv[-1])
    print(f'{p = } {save = } {n = }')
    get_image(p, save, n=n)


if __name__ == "__main__":
    main()
