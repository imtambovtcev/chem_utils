import sys
from pathlib import Path
from typing import List, Optional, Union


def xyz_str_to_allxyz_str(content: str) -> str:
    """
    Process the content of an XYZ file and return the modified content.
    
    Parameters:
    - content: Content of the XYZ file as a string.
    
    Returns:
    - Modified content as a string.
    """
    n = content.partition('\n')[0]
    lines = content.split('\n')
    lines = ['>\n'+line if line == n else line for line in lines][1:]
    processed_content = n+'\n'+'\n'.join(lines)
    return processed_content

def xyz_to_allxyz(input_files: Union[List[Union[Path, str]], Union[Path, str]], output: Optional[Union[Path, str]] = None) -> None:
    """
    Process a list of XYZ files or a single XYZ file and save the results to corresponding .allxyz files.
    
    Parameters:
    - input_files: List of paths or a single path (either Path objects or strings) pointing to XYZ files to be processed.
    - output: Optional path or string for the output file. If not provided for each input file, defaults to replacing the `.xyz` extension with `.allxyz`.
    """
    # Ensure input_files is a list
    if not isinstance(input_files, list):
        input_files = [input_files]
    
    # Convert all inputs to Path objects
    input_paths = [Path(d) for d in input_files]
    
    for file in input_paths:
        with open(file, "r") as f:
            content = f.read()
        
        processed_content = xyz_str_to_allxyz_str(content)
        
        if output is None:
            output_path = file.with_suffix('.allxyz')
        else:
            output_path = Path(output)
        
        with open(output_path, "w") as f:
            f.write(processed_content)


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
