from __future__ import annotations

import argparse
import pathlib
from .path import Path
from .molecule import Molecule


def render_molecule_from_files(filenames, alpha=1.0, interactive=False, as_path=False):
    savename = pathlib.Path(filenames[0]).stem
    files = [Path.load(f) for f in filenames]  # Load files
    # Flatten list of images
    images = [image for path in files for image in path]

    # Adjust alpha to match the length of images
    if isinstance(alpha, (int, float)):
        alpha = [alpha] * len(images)
    elif len(alpha) == 1:
        alpha = alpha * len(images)
    else:
        assert len(alpha) == len(
            images), f"Alpha values must be the same length as the number of images or 1: {len(alpha)} != {len(images)}"

    if as_path:
        path = Path(images)
        if interactive:
            path.render(alpha=alpha, show=True)
        else:
            path.render(save=savename, alpha=alpha, show=False)
    else:
        # Case for one image
        if len(images) == 1:
            if interactive:
                images[0].render(alpha=alpha[0], show=True)
            else:
                images[0].render(save=savename, alpha=alpha[0], show=False)
        # Case for two images
        elif len(images) == 2:
            plotter = images[0].render(alpha=alpha[0], show=False)
            if interactive:
                images[1].render(plotter=plotter, alpha=alpha[1], show=True)
            else:
                images[1].render(plotter=plotter, save=savename,
                                 alpha=alpha[1], show=False)
        # Case for three or more images
        else:
            plotter = images[0].render(alpha=alpha[0], show=False)

            for i, image in enumerate(images[1:-1], start=1):
                image.render(plotter=plotter, alpha=alpha[i], show=False)

            if interactive:
                images[-1].render(plotter=plotter, alpha=alpha[-1], show=True)
            else:
                images[-1].render(plotter=plotter, save=savename,
                                  alpha=alpha[-1], show=False)


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('inputs', metavar='I', type=str, nargs='*', default=['./'],
                        help='an input directory or file for processing')
    parser.add_argument('--alpha', type=float, nargs='+', default=[1.0],
                        help='the alpha value(s) for rendering, default is 1.0')
    parser.add_argument('--as_path', action='store_true',
                        help='treat multiple images as path')
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
    render_molecule_from_files(
        inp, alpha=args.alpha, interactive=args.interactive, as_path=args.as_path)


if __name__ == "__main__":
    main()
