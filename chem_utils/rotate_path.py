import sys
import pathlib
from ase import Atoms
from ase.io import read, write

from .motor import Motor, Path


def rotate_path(input_path, settings=None):
    """
    Rotates a given path and returns/saves it.
    """
    mode = set()

    # Detect type of input
    if isinstance(input_path, list):
        path = Path([Motor(atoms) for atoms in input_path])
        mode.add('return')
    elif isinstance(input_path, Atoms):
        path = Path([Motor(input_path)])
        mode.add('return')
    else:
        atoms_list = read(str(input_path), index=':')
        path = Path([Motor(atoms) for atoms in atoms_list])
        mode.add('save')

    path.rotate()

    return_path = path.copy()

    if 'return' in mode:
        return return_path
    if 'save' in mode:
        return_path.save(str(input_path).replace('.xyz', '_rotated.xyz'))


def main():
    _input = ['./'] if len(sys.argv) <= 1 else sys.argv[1:]
    paths = [pathlib.Path(d) for d in _input]

    files_to_process = []
    for path in paths:
        if path.is_dir():
            files_to_process.extend(path.glob('*.xyz'))
        else:
            files_to_process.append(path)

    for p in files_to_process:
        rotate_path(p)


if __name__ == "__main__":
    main()
