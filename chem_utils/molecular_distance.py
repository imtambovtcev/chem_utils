from pathlib import Path
from ase.io import read
import numpy as np
import sys
from scipy.spatial.distance import cdist
import argparse


def atoms_distance(atoms_a, atoms_b, savetxt=None, return_position=False, print_diff=True):
    assert len(atoms_a.get_chemical_symbols()) == len(
        atoms_b.get_chemical_symbols()), "Mismatch in number of atoms"

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


def load_atoms(file, image_index=None):
    """Load atoms from a given file, optionally using an image index if applicable."""
    if file.suffix == '.gpw':
        from gpaw import restart
        atoms = restart(str(file), txt=None)[0]
    else:
        atoms = read(str(file), index=image_index)
    return atoms


def main():
    parser = argparse.ArgumentParser(
        description="Compare atomic distances between two structures.")
    parser.add_argument('file_a', type=Path,
                        help="Path to the first atomic structure file.")
    parser.add_argument('file_b', type=Path,
                        help="Path to the second atomic structure file.")
    parser.add_argument('--index_a', type=int, default=None,
                        help="Image index for atoms_a if the file contains multiple images (default: None).")
    parser.add_argument('--index_b', type=int, default=None,
                        help="Image index for atoms_b if the file contains multiple images (default: None).")
    parser.add_argument('--savetxt', type=str, default=None,
                        help="Optional path to save the distance difference as a text file.")
    parser.add_argument('--print_diff', action='store_true', default=True,
                        help="Print the max difference between atomic distances.")

    args = parser.parse_args()

    # Load atoms with optional index
    atoms_a = load_atoms(args.file_a, args.index_a)
    atoms_b = load_atoms(args.file_b, args.index_b)

    # Compute and display distances
    atoms_distance(atoms_a, atoms_b, args.savetxt, print_diff=args.print_diff)


if __name__ == "__main__":
    main()
