from pathlib import Path
from ase.io import read
import numpy as np
import sys
from scipy.spatial.distance import cdist

sys.path.insert(1, '/users/home/imtambovtsev/')


def atoms_distance(atoms_a, atoms_b, savetxt=None, return_position=False, print_diff=True):
    assert len(atoms_a.get_chemical_symbols()) == len(
        atoms_b.get_chemical_symbols())
    diff = (cdist(atoms_a.get_positions(), atoms_a.get_positions())
            - cdist(atoms_b.get_positions(), atoms_b.get_positions()))
    if savetxt is not None:
        np.savetxt(savetxt, diff)
    if print_diff:
        print(f'{diff.max() = }')
    if return_position:
        return diff.max(), np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    else:
        return diff.max()


def main():
    file_a = Path(sys.argv[1])
    file_b = Path(sys.argv[2])
    savetxt = None if len(sys.argv) <= 3 or sys.argv[3] in [
        'None', 'none'] else sys.argv[3]

    if file_a.suffix == '.gpw':
        from gpaw import restart
        atoms_a = restart(str(file_a), txt=None)[0]
    else:
        atoms_a = read(str(file_a))

    if file_b.suffix == '.gpw':
        from gpaw import restart
        atoms_b = restart(str(file_b), txt=None)[0]
    else:
        atoms_b = read(str(file_b))

    atoms_distance(atoms_a, atoms_b, savetxt)

if __name__ == "__main__":
    main()
