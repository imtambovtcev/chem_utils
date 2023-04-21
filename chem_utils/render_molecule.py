import sys
import pyvista as pv
import numpy as np
import ase
from ase.io import read
from ase.geometry.analysis import Analysis
import pandas as pd

from pathlib import Path


# from .rotate_path import rotate_path

default_atoms_settings = pd.read_csv(
    str(Path(__file__).parent/'pyvista_render_settings.csv'))
default_atoms_settings.index = list(default_atoms_settings['Name'])
default_atoms_settings['Color'] = [[int(i) for i in s.replace(
    '[', '').replace(']', '').split(',')] for s in default_atoms_settings['Color']]


def render_molecule(plotter: pv.Plotter, atoms: ase.Atoms, atoms_settings=None, show_hydrogens=True, alpha=1.0, atom_numbers=False, show_hydrogen_bonds=True):

    if atoms_settings is None:
        atoms_settings = default_atoms_settings
    symb = list(set(atoms.symbols))
    # makes atom settings search bit faster
    _atoms_settings = atoms_settings.loc[symb]
    # render atoms
    for atom in atoms:
        symbol = atom.symbol if atom.symbol in _atoms_settings.index else 'Unknown'
        if show_hydrogens or symbol != 'H':  # hydrogen show check
            sphere = pv.Sphere(
                radius=_atoms_settings.loc[symbol]['Radius'], center=atom.position, theta_resolution=12, phi_resolution=12)
            plotter.add_mesh(
                sphere, color=_atoms_settings.loc[symbol]['Color'], smooth_shading=True, opacity=alpha)

    if atom_numbers:
        poly = pv.PolyData(atoms.positions)
        poly["My Labels"] = [str(i) for i in range(poly.n_points)]
        plotter.add_point_labels(
            poly, "My Labels", point_size=20, font_size=36)

    # start_time = time.perf_counter()
    # render bonds
    pairs = [(a, b) for idx, a in enumerate(symb) for b in symb[idx + 1:]]+[(i, i)
                                                                            for i in symb] if len(symb) > 1 else [(symb[0], symb[0])]

    ana = Analysis(atoms)

    if show_hydrogens and show_hydrogen_bonds:
        # print(ana.get_bonds('F', 'H', unique=True))
        h_coords = atoms.get_positions(
        )[[a == 'H' for a in atoms.get_chemical_symbols()]]
        f_coords = atoms.get_positions(
        )[[a == 'F' for a in atoms.get_chemical_symbols()]]
        for h_coord in h_coords:
            for f_coord in f_coords:
                if np.linalg.norm(h_coord-f_coord) < 2.2:
                    line = pv.Line(h_coord, f_coord)
                    plotter.add_mesh(line, color='white',
                                     opacity=alpha, line_width=2)

    for bond_type in pairs:

        if show_hydrogens or not ('H' in bond_type):  # hydrogen show check
            bonds = ana.get_bonds(*bond_type, unique=True)[0]

            for bond in bonds:
                # print(bond)
                atom_a = atoms[bond[0]].position
                atom_b = atoms[bond[1]].position
                cylinder = pv.Cylinder(center=0.5*(atom_a+atom_b), direction=atom_b -
                                       atom_a, height=np.linalg.norm(atom_b-atom_a), radius=0.05, resolution=10)
                plotter.add_mesh(cylinder, color='#D3D3D3',
                                 smooth_shading=True, opacity=alpha)


def render_molecule_from_path(path, save=False):
    for i, atoms in enumerate(path):
        p = pv.Plotter(notebook=True)
        p.set_background('black')
        render_molecule(plotter=p, atoms=atoms,
                        atoms_settings=default_atoms_settings)
        # p.view_vector((-1, 0, 0), (0, 1, 0))
        if save:
            p.show(screenshot=save, window_size=[1000, 1000])
        else:
            return p


def render_molecule_from_file(filename, save=None):
    path = read(filename, index=':')
    s = filename[:-4] + '_{}.png'.format(
        i) if save is None else save[:-4]+'_{}'.format(i) + save[-4:]
    render_molecule_from_atoms(path, s)


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
    [render_molecule_from_file(str(p)) for p in input]


if __name__ == "__main__":
    main()
