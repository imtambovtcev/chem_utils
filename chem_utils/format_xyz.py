import sys
from pathlib import Path


def format_xyz(string, keep_comment_line=False):
    """
    Formats an XYZ string according to the following rules:
    - Detect if the first line contains the number of atoms (a single integer).
    - If yes, treat the second line as a comment and keep it optionally based on `keep_comment_line`.
    - If no integer is found in the first line, assume the first line is an atom and process accordingly.
    - Remove empty or whitespace-only lines from the atom list.
    - Ensure that the number of atom lines matches the number specified, if applicable.
    - Validate that each atom line contains valid float numbers in positions 2-4 (x, y, z coordinates).
    - Format the output as: number of atoms, comment line (optional), atom lines.

    Parameters:
    - string (str): The input XYZ content.
    - keep_comment_line (bool): Whether to keep the comment line if present.

    Returns:
    - final_string (str): The formatted XYZ content.
    """
    lines = string.strip().split(
        '\n')  # Split the input by line and remove leading/trailing spaces
    lines = [line.strip()
             for line in lines if line.strip()]  # Clean out empty lines

    # Check if the first line is a single integer (number of atoms)
    try:
        num_atoms = int(lines[0])
        comment_line = lines[1] if keep_comment_line else ""
        atom_lines = lines[2:]  # Atoms start from the third line
    except ValueError:
        # If not an integer, treat the input as having atom lines starting from the first line
        num_atoms = len(lines)
        comment_line = "" if not keep_comment_line else None  # No comment line in this case
        atom_lines = lines

    # Verify and clean atom lines
    cleaned_atom_lines = []
    for i, line in enumerate(atom_lines):
        # Split the line and validate the first 4 elements are valid (atom + 3 coordinates)
        elements = line.split()
        assert len(elements) >= 4, f"Line {
            i + 1} does not have enough data: {line}"
        # Ensure that positions 2-4 are floats (x, y, z coordinates)
        try:
            coords = list(map(float, elements[1:4]))
        except ValueError:
            raise ValueError(
                f"Line {i + 1} contains invalid coordinates: {elements[1:4]}")
        # Keep the first 4 values only (atom + coordinates)
        cleaned_atom_lines.append('    '.join(elements[:4]))

    # Assert the number of atom lines matches the specified number of atoms
    assert len(cleaned_atom_lines) == num_atoms, (
        f"Expected {num_atoms} atom lines, but got {len(cleaned_atom_lines)}"
    )

    # Format the final XYZ string
    final_string = f"{num_atoms}\n"
    if comment_line is not None:
        final_string += f"{comment_line}\n"
    else:
        final_string += "\n"
    final_string += '\n'.join(cleaned_atom_lines) + '\n'

    print(final_string)
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


def main():
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


if __name__ == "__main__":
    main()
