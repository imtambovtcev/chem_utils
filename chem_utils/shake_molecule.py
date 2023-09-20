import sys
import pathlib
from .motor import Molecule

def format_all_xyz(input):
    for file in input:
        m = Molecule.load(file)
        m.shake()
        m.save(file)


def main():
    _input = ['./'] if len(sys.argv) <= 1 else sys.argv[1:]
    print(_input)
    _input = [pathlib.Path(d) for d in _input]
    input = []
    for d in _input:
        if d.is_dir():
            add = d.glob('*.xyz')
            input.extend(add)
        else:
            input.append(d)
    print(input)
    format_all_xyz(input)


if __name__ == "__main__":
    main()
