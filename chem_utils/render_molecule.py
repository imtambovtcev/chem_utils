from __future__ import annotations

import argparse
import pathlib
from .path import Path


def render_molecule_from_file(filename, alpha=1.0, interactive=False):
    path = Path.load(filename)
    if len(path) == 1:
        path = path[0]
    if interactive:
        path.render(alpha=alpha, show=True)
    else:
        path.render(save=str(filename)[:-4], alpha=alpha, show=False)


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('inputs', metavar='I', type=str, nargs='*', default=['./'],
                        help='an input directory or file for processing')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='the alpha value for rendering')
    # Add interactive argument as a flag
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='enable interactive mode for rendering')

    args = parser.parse_args()

    _input = args.inputs
    print(_input)
    _input = [pathlib.Path(i) for i in _input]
    inp = []
    for i in _input:
        if i.is_dir():
            add = i.glob('*.xyz')
            inp.extend(add)
        elif i.is_file():
            inp.append(i)
    print(inp)

    # Use the interactive flag from args when calling render_molecule_from_file
    [render_molecule_from_file(
        str(p), alpha=args.alpha, interactive=args.interactive) for p in inp]


if __name__ == "__main__":
    main()
