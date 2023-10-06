import sys
from pathlib import Path
import ase
from ase.io import read
from ase.io import write
from ase.geometry.analysis import Analysis
import numpy as np
from .format_xyz import format_xyz_file
from .xyz_to_allxyz import xyz_to_allxyz
from pathlib import Path


def linear_interploation(atoms_a: ase.Atoms, atoms_b: ase.Atoms, n=1):
    atoms = []
    for i in range(n):
        atoms.append(atoms_a.copy())
        atoms[-1].set_positions(atoms_a.get_positions()+((i+1)/(n+1))
                                * (atoms_b.get_positions()-atoms_a.get_positions()))

    atoms
    return atoms


def holes(l):
    holes_list = []
    in_hole = False
    for i, value in enumerate(l):
        if value:
            in_hole = False
        else:
            if in_hole:
                holes_list[-1][1] += 1
            else:
                in_hole = True
                holes_list.append([i, 1])

    return holes_list


def diff(list1, list2):
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    return list(c - d)


def find_holes(path):
    """returns list of bool True in no brond broke from image 0 on path False if something is broken"""
    ana = Analysis(path[0])
    CCBonds_cis = ana.get_bonds('C', 'C', unique=True)[0]
    CHBonds_cis = ana.get_bonds('C', 'H', unique=True)[0]

    print(CCBonds_cis)
    # print(CHBonds_cis)

    bonds = []

    for i, atoms in enumerate(path):
        # print(i)

        ana = Analysis(atoms)
        CCBonds = ana.get_bonds('C', 'C', unique=True)[0]
        CHBonds = ana.get_bonds('C', 'H', unique=True)[0]

        CCdiff = diff(CCBonds_cis, CCBonds)
        CHdiff = diff(CHBonds_cis, CHBonds)

        if len(CCdiff) == 0 and len(CHdiff) == 0:
            bonds.append(True)
        else:
            bonds.append(False)

            print(f'At image {i} the difference in bonding was found:')
            if len(CCdiff) != 0:
                print(f'C-C bonds {CCdiff} are diffrent')
            if len(CHdiff) != 0:
                print(f'C-H bonds {CHdiff} are diffrent')

    assert len(bonds) == len(path)  # check file

    assert bonds[0]  # check file
    assert bonds[-1]  # initial and final has different bonds

    return bonds


def check_and_copy_path(pathname, save_file):
    _pathname = Path(pathname)

    path = read(str(_pathname), index=':')
    print(path[0])
    print(f'{len(path) = }')

    bonds = find_holes(path)

    print(bonds)

    if len(holes) > 0:
        for hole, length in holes(bonds):
            print(f'{hole = } {length = }')
            path[hole:hole+length] = linear_interploation(
                path[hole-1], path[hole+length], n=length)

        bonds = find_holes(path)
        print(bonds)

    write(str(save_file), path)
    format_xyz_file(str(save_file))
    xyz_to_allxyz([Path(save_file)])


def main():
    pathname, save_file = sys.argv[1]
    check_and_copy_path(pathname, save_file)


if __name__ == "__main__":
    main()
