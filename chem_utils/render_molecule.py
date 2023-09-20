from __future__ import annotations

import argparse
import pathlib
from .motor import Path


def render_molecule_from_file(filename, alpha=1.0):
    path = Path.load(filename)
    path.render(save=str(filename)[:-4], alpha=alpha)


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('inputs', metavar='I', type=str, nargs='*', default=['./'],
                        help='an input directory or file for processing')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='the alpha value for rendering')
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
    [render_molecule_from_file(str(p), alpha=args.alpha) for p in inp]


if __name__ == "__main__":
    main()
