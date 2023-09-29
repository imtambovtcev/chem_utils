# Chem Utils

2. Navigate to the cloned repository:

   ```shell
   cd chem_utils
   ```
3. Install the package in editable mode:
   ```
   pip install . -e
   ```

## Modules

- `molecule.py`: Represents and manipulates molecular structures.
- `motor.py`: Represents a Motor, extending the functionality of the `Molecule` class.
- `fragment.py`: Represents a molecular fragment, extending the functionality of the `Molecule` class.
- `eldens.py`: Deals with the representation and manipulation of electron density data.
- `path.py`: Represents a sequence or collection of molecular configurations (images).

- `frequency.py`: Represents and manipulates vibrational frequencies and modes.

- `utils.py`: Provides various utility functions.
- `xyz_from_allxyz.py`: Contains functions to extract XYZ coordinates from a file containing multiple XYZ coordinate sets.