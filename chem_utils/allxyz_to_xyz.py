import numpy as np
import sys
from pathlib import Path


def allxyz_to_xyz(input):
    for file in input:
        f = open(file, "r")
        s = f.read()
        s = s.replace('>\n', '')
        f.close()
        f = open(file.parent / (file.stem + '.xyz'), "w")
        f.write(s)
        f.close()


def main():
    _input = ['./'] if len(sys.argv) <= 1 else sys.argv[1:]
    print(_input)
    _input = [Path(d) for d in _input]
    input = []
    for d in _input:
        if d.is_dir():
            add = d.glob('*.allxyz')
            input.extend(add)
        else:
            input.append(d)
    print(input)
    allxyz_to_xyz(input)


if __name__ == "__main__":
    main()
