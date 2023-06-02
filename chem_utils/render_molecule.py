import sys
import pyvista as pv
import numpy as np
import ase
from ase.io import read
from ase.geometry.analysis import Analysis
import pandas as pd

from pathlib import Path
import argparse


# from .rotate_path import rotate_path

default_atoms_settings = pd.read_csv(
    str(Path(__file__).parent/'pyvista_render_settings.csv'))
default_atoms_settings.index = list(default_atoms_settings['Name'])
default_atoms_settings['Color'] = [[int(i) for i in s.replace(
    '[', '').replace(']', '').split(',')] for s in default_atoms_settings['Color']]


def render_molecule(plotter: pv.Plotter, atoms: ase.Atoms, atoms_settings=None, show_hydrogens=True, alpha=1.0, atom_numbers=False, show_hydrogen_bonds=False):

    if atoms_settings is None:
        atoms_settings = default_atoms_settings

    symb = list(set(atoms.symbols))
    _atoms_settings = atoms_settings.loc[symb]
    atom_positions = atoms.positions
    atom_symbols = atoms.get_chemical_symbols()

    for atom, position, symbol in zip(atoms, atom_positions, atom_symbols):
        symbol = symbol if symbol in _atoms_settings.index else 'Unknown'
        if show_hydrogens or symbol != 'H':
            sphere = pv.Sphere(
                radius=_atoms_settings.loc[symbol]['Radius'], center=position, theta_resolution=12, phi_resolution=12)
            plotter.add_mesh(
                sphere, color=_atoms_settings.loc[symbol]['Color'], smooth_shading=True, opacity=alpha)

    if atom_numbers:
        poly = pv.PolyData(atom_positions)
        poly["My Labels"] = [str(i) for i in range(poly.n_points)]
        plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)

    pairs = [(a, b) for idx, a in enumerate(symb) for b in symb[idx + 1:]]+[(i, i) for i in symb] if len(symb) > 1 else [(symb[0], symb[0])]
    ana = Analysis(atoms)

    if show_hydrogens and show_hydrogen_bonds:
        h_coords = atom_positions[np.array(atom_symbols) == 'H']
        f_coords = atom_positions[np.array(atom_symbols) == 'F']

        # Compute the distance matrix
        distance_matrix = np.linalg.norm(h_coords[:, None, :] - f_coords[None, :, :], axis=-1)
        # Iterate over the pairs within distance threshold
        for h_coord, f_coord in zip(h_coords[distance_matrix < 2.2], f_coords[distance_matrix < 2.2]):
            line = pv.Line(h_coord, f_coord)
            plotter.add_mesh(line, color='white', opacity=alpha, line_width=2)

    for bond_type in pairs:
        if show_hydrogens or not ('H' in bond_type):
            bonds = ana.get_bonds(*bond_type, unique=True)[0]

            for bond in bonds:
                atom_a = atom_positions[bond[0]]
                atom_b = atom_positions[bond[1]]
                atom_a_radius = _atoms_settings.loc[atom_symbols[bond[0]]]['Radius']
                atom_b_radius = _atoms_settings.loc[atom_symbols[bond[1]]]['Radius']

                bond_vector = atom_b - atom_a
                bond_length = np.linalg.norm(bond_vector)
                unit_vector = bond_vector / bond_length

                atom_a_adjusted = atom_a + unit_vector * atom_a_radius
                atom_b_adjusted = atom_b - unit_vector * atom_b_radius
                bond_length_adjusted = bond_length - atom_a_radius - atom_b_radius

                cylinder = pv.Cylinder(center=0.5*(atom_a_adjusted + atom_b_adjusted),
                                       direction=bond_vector,
                                       height=bond_length_adjusted,
                                       radius=0.05,
                                       resolution=10)
                plotter.add_mesh(cylinder, color='#D3D3D3', smooth_shading=True, opacity=alpha)



def render_molecule_from_atoms(atoms, plotter=None, save=None, cpos=None, atoms_settings=default_atoms_settings, alpha=1.0, notebook=False, auto_close=True, interactive=False, background_color='black', show_hydrogen_bonds=False):
    if plotter is None:
        plotter = pv.Plotter(notebook=notebook)
        plotter.set_background(background_color)
    render_molecule(plotter=plotter, atoms=atoms,
                    atoms_settings=atoms_settings, alpha=alpha, show_hydrogen_bonds=show_hydrogen_bonds)
    if save is not None:
        plotter.show(screenshot=save, window_size=[
                     1000, 1000], cpos=cpos, auto_close=auto_close, interactive=interactive)
    else:
        return plotter


def render_molecule_from_path(path, plotter=None, save=None, cpos=None, atoms_settings=default_atoms_settings, alpha=1.0):
    if save is None:
        save = [None for _ in path]
    if plotter is None:
        plotter = [None for _ in path]
    for i, (atoms, s, p) in enumerate(zip(path, save, plotter)):
        if p is None:
            p = pv.Plotter(notebook=True)
            p.set_background('black')

        render_molecule(plotter=p, atoms=atoms,
                        atoms_settings=atoms_settings, alpha=alpha)
        # p.view_vector((-1, 0, 0), (0, 1, 0))
        if s is not None:
            p.show(screenshot=s, window_size=[1000, 1000], cpos=cpos)
        # else:
        #     return p


def render_molecule_from_file(filename, save=None, alpha=1.0):
    path = read(filename, index=':')
    if save is None:
        save = [filename[:-4] + f'_{i}.png' for i in range(len(path))]
    render_molecule_from_path(path=path, save=save, alpha=alpha)


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('inputs', metavar='I', type=str, nargs='*', default=['./'],
                        help='an input directory or file for processing')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='the alpha value for rendering')
    args = parser.parse_args()

    _input = args.inputs
    print(_input)
    _input = [Path(i) for i in _input]
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
