import numpy as np
import sys
from pathlib import Path


def xyz_to_allxyz(input):
    for file in input:
        f = open(file, "r")
        s = f.read()
        n = s.partition('\n')[0]
        s = s.split('\n')
        s = ['>\n'+st if st == n else st for st in s][1:]
        s = n+'\n'+'\n'.join(s)
        print('nimages = {}'.format(len(s.split('>\n'))))

        f.close()
        f = open(file.parent / (file.stem + '.allxyz'), "w")
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
    xyz_to_allxyz(input)


if __name__ == "__main__":
    main()
