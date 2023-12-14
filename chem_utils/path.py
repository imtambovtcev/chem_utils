from __future__ import annotations

import warnings
from ase import Atoms
from ase.io import read, write

from .molecule import Molecule
from .motor import Motor


class Path:
    def __init__(self, images=None):
        self.images = []
        if images is not None:
            # Check if any image is of the Motor class
            contains_motor = any(isinstance(image, Motor) for image in images)
            if contains_motor:
                # Convert all images to Motor class
                converted_images = [Motor(image) if not isinstance(
                    image, Motor) else image for image in images]
            else:
                # Convert all images to Molecule class
                converted_images = [Molecule(image) if not isinstance(
                    image, Molecule) else image for image in images]

            # Check that all images satisfy __eq__ requirements
            reference_image = converted_images[0]
            for idx, img in enumerate(converted_images[1:], start=1):
                if reference_image != img:
                    warnings.warn(
                        f"The image at index {idx} atoms are different.")

            # Initialize the self.images attribute
            self.images = converted_images

    def _get_type(self):
        """Retrieve the type of the images in the path."""
        return type(self[0]) if len(self.images) > 0 else None

    def _convert_type(self, image):
        """Convert the image to the type of images in the path."""
        if len(self.images) == 0:
            if not isinstance(image, Molecule):
                return Molecule(image)
            return image
        current_type = self._get_type()
        # print(f'{current_type = }')
        return current_type(image)

    def to_type(self, new_type):
        """
        Convert all images in the path to the specified type.

        Parameters:
        - new_type (type): The desired type to which all images should be converted.
        """
        if not isinstance(new_type, type):
            raise TypeError(
                f"Expected 'new_type' to be a type, got {type(new_type)}")

        self.images = [new_type(image) for image in self.images]

    def __getattr__(self, attr):
        """
        If the attribute (method in this context) is not found, 
        this method is called.
        """
        def method(*args, **kwargs):
            return [getattr(image, attr)(*args, **kwargs) for image in self.images]

        return method

    def __getitem__(self, index):
        """Retrieve an image at the specified index."""
        return self.images[index]

    def __setitem__(self, index, image):
        """Replace an image at the specified index."""

        # Handle single integer index
        if isinstance(index, int):
            converted_image = self._convert_type(image)
            # Check for equality with the first image in the path
            if len(self.images) > 0 and self.images[0] != converted_image:
                warnings.warn(
                    f"The provided image does not satisfy the equality requirements with the first image in the path.")
            self.images[index] = converted_image
            return

        # Handle slices
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.images))
            if not isinstance(image, (list, tuple)):
                raise ValueError(
                    "For slice assignment, the image must be a list or tuple of images.")
            for i, img in zip(range(start, stop, step), image):
                self[i] = img
            return

        # Handle lists or other iterables
        if isinstance(index, (list, tuple)):
            if not isinstance(image, (list, tuple)) or len(index) != len(image):
                raise ValueError(
                    "For list or tuple assignment, the image must be a list or tuple of images with the same length as the index.")
            for i, img in zip(index, image):
                self[i] = img
            return

        raise TypeError(f"Index of type {type(index)} is not supported.")

    def __iter__(self):
        """Make the class iterable."""
        return iter(self.images)

    def __len__(self):
        """Return the number of images in the list."""
        return len(self.images)

    def append(self, image):
        """Add a new image to the list."""
        # Convert the image to the appropriate type
        converted_image = self._convert_type(image)

        # Check if the image satisfies __eq__ requirements with the existing images
        if self.images and not all(img == converted_image for img in self.images):
            warnings.warn(
                "The appended image does not satisfy the equality requirements with existing images in the path.")

        self.images.append(converted_image)

    def remove_image(self, index):
        """Remove an image from the list at the specified index."""
        del self.images[index]

    def save(self, filename):
        """Save all images in the list to a file."""
        write(filename, self.images)

    @classmethod
    def load(cls, filename):
        """Load images from a file and return a new Path object with those images."""
        return cls(read(filename, index=':'))

    def rotate(self):
        assert isinstance(self[0], Motor)
        zero_atom, x_atom, no_z_atom = self[0].find_rotation_atoms()
        for motor in self:
            motor.to_origin(zero_atom, x_atom, no_z_atom)

    def reorder_atoms_of_intermediate_images(self):
        half_len = len(self) // 2

        for i in range(1, half_len + 1):
            mapping = self[i - 1].best_mapping(self[i])
            if mapping is None:
                raise ValueError(
                    f"No valid mapping found for molecules at index {i} and {i-1}.")
            self[i] = self[i].reorder(mapping)

        # Adjust from the end towards the center
        for i in range(len(self) - 2, half_len - 1, -1):
            mapping = self[i + 1].best_mapping(self[i])
            if mapping is None:
                raise ValueError(
                    f"No valid mapping found for molecules at index {i} and {i+1}.")
            self[i] = self[i].reorder(mapping)

    def find_bonds_breaks(self):
        assert self[0].get_all_bonds() == self[-1].get_all_bonds()
        first_image_bonds = self[0].get_all_bonds()
        return [first_image_bonds == image.get_all_bonds() for image in self]

    def find_holes(self):
        l = self.find_bonds_breaks()
        holes_list = []
        in_hole = False
        for i, value in enumerate(l):
            if value:
                in_hole = False
            else:
                if in_hole:
                    holes_list[-1][1] += 1
                else:
                    in_hole = True
                    holes_list.append([i, 1])
        return holes_list

    def fix_bonds_breaks(self):
        l = self.find_holes()

        for hole, length in l:
            # print(f'{hole = } {length = }')
            self[hole:hole+length] = self[hole -
                                          1].linear_interploation(self[hole+length], n=length).images

        l = self.find_holes()

        if len(l) > 0:
            warnings.warn(
                f'Linear interpolation didn\'t fix the problem. Holes:{l}')
            for hole, length in l:
                # print(f'{hole = } {length = }')
                for i in range(hole, hole+length):
                    p = self[hole-1].copy()
                    self[i] = p

        l = self.find_holes()

        if len(l) > 0:
            warnings.warn(
                f'Problem wasn\'t fixed. Holes:{l}')

    def to_xyz_string(self):
        return "".join([image.to_xyz_string() for image in self])

    def save_as_xyz(self, filename):
        with open(filename, "w") as text_file:
            text_file.write(self.to_xyz_string())

    def copy(self):
        return Path([image.copy() for image in self])

    def to_allxyz_string(self):
        return ">\n".join([image.to_xyz_string() for image in self])

    def save_as_allxyz(self, filename):
        with open(filename, "w") as text_file:
            text_file.write(self.to_allxyz_string())

    def save(self, filename):
        if filename.endswith('_all.xyz'):
            self.save_as_allxyz(filename)
        elif filename.endswith('.xyz'):
            self.save_as_xyz(filename)
        else:
            raise ValueError("Unsupported file format")

    def __str__(self):
        info = [
            f"Path Information:",
            f"Number of Images: {len(self)}",
            f"Image Type: {self._get_type().__name__ if self._get_type() else 'None'}",
            f"First Image:\n{self[0] if len(self) > 0 else 'None'}",
            f"Last Image:\n{self[-1] if len(self) > 0 else 'None'}",
        ]

        return "\n".join(info)

    def render(self, save=None, **kwargs):
        if save:
            for i, image in enumerate(self):
                image.render(save=f'{save}_{i}.png', **kwargs)

        else:
            current_idx = 0
            p = self[current_idx].render()
            print(
                "Press 'n' to move to the next molecule, 'p' to go back, and 'q' to quit.")

            def key_press(obj, event):
                nonlocal current_idx
                key = obj.GetKeySym()  # Get the pressed key
                if key == 'n' and current_idx < len(self) - 1:
                    current_idx += 1
                elif key == 'p' and current_idx > 0:
                    current_idx -= 1
                elif key == 'q':
                    # Exit the rendering
                    p.close()
                    return
                # Update the rendered molecule based on the current_idx
                p.clear()
                self[current_idx].render(p)
                p.reset_camera()
                p.render()

            p.iren.add_observer('KeyPressEvent', key_press)
            p.show()

    def render_alongside(self, other, alpha=1.0):
        current_idx = 0
        p = self[current_idx].render()

        if isinstance(other, Path):
            other[current_idx].render(p, alpha=alpha)
            print(
                "Press 'n' to move to the next pair of molecules, 'p' to go back, and 'q' to quit.")

            def key_press(obj, event):
                nonlocal current_idx
                key = obj.GetKeySym()  # Get the pressed key
                if key == 'n' and current_idx < min(len(self), len(other)) - 1:
                    current_idx += 1
                elif key == 'p' and current_idx > 0:
                    current_idx -= 1
                elif key == 'q':
                    # Exit the rendering
                    p.close()
                    return
                # Update the rendered molecules based on the current_idx
                p.clear()
                self[current_idx].render(p)
                other[current_idx].render(p, alpha=alpha)
                p.reset_camera()
                p.render()

            p.iren.add_observer('KeyPressEvent', key_press)
        elif isinstance(other, Atoms):
            other.render(p, alpha=alpha)
        else:
            raise TypeError(
                f"Expected other to be of type Path or ase.Atoms, got {type(other)}")

        p.show()
