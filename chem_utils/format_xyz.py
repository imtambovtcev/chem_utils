import numpy as np
import sys
from pathlib import Path


def format_xyz(string):
    lines = string.split('\n')
    d = [len(l.split()) for l in lines]
    # print(d)
    if d[0] == 1:
        final_string = '\n'.join(lines[0:2]) + '\n'
        start_line = 2
    else:
        final_string = '{}\n\n'.format(np.sum(np.array(d) > 0))
        start_line = 0
    lines = lines[start_line:]

    for i, l in enumerate(filter(None, lines)):
        final_string += '        '.join(l.split()[:4])
        if i <= len(lines):
            final_string += '\n'
    return final_string


def format_xyz_file(filename):
    with open(filename, 'r') as file:
        final_string = format_xyz(file.read())
    with open(filename, 'w') as file:
        file.write(final_string)


def format_all_xyz(input):
    for file in input:
        f = open(file, "r")
        s = format_xyz(f.read())
        f.close()
        f = open(file, "w")
        f.write(s)
        f.close()


if __name__ == "__main__":
    _input = ['./'] if len(sys.argv) <= 1 else sys.argv[1:]
    print(_input)
    _input = [Path(d) for d in _input]
    input = []
    for d in _input:
        if d.is_dir():
            add = d.glob('*.xyz')
            input.extend(add)
        else:
            input.append(d)
    print(input)
    format_all_xyz(input)
