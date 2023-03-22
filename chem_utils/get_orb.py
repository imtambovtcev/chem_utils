import os
import sys

import numpy as np

from ase.io import read, write
from ase.parallel import paropen
import ase.units as units

from gpaw import GPAW, restart
from gpaw.mpi import world, rank, MASTER

import sys
from pathlib import Path


def gen_cubes(atoms, calc, name, i=0, a=0, spins=1):
    assert (isinstance(i, int) == True and i <= 0)
    assert (isinstance(a, int) == True and a >= 0)

    # Write out orbitals as cube files and find isovalue
    # for plotting based on cutoff criteria
    dv = calc.wfs.gd.dv * units.Bohr**3
    max_isoval = 0.5

    # For an Aufbau distribution of the occupation numbers
    # nhomo is the index of the homo and nlumo is the index
    # of the lumo. Prints the orbitals from nhomo+i to nlumo+a
    # for s in range(calc.wfs.nspins):
    for s in range(spins):
        f_n = calc.wfs.kpt_u[s].f_n
        nlumo = len(f_n[f_n > 0])
        nhomo = nlumo - 1

        for n in range(nhomo+i, nlumo+a+1):
            print(name+' orbital %d spin %d' % (n, s))
            orb = calc.get_pseudo_wave_function(band=n, spin=s)
            write(name+'_%d_spin%d.cube' % (n, s), atoms, data=orb)


if __name__ == "__main__":
    file = sys.argv[1]
    file = Path(file)
    i = -5 if len(sys.argv) <= 2 else int(sys.argv[2])
    a = 5 if len(sys.argv) <= 3 else int(sys.argv[3])
    spins = 1 if len(sys.argv) <= 3 else int(sys.argv[4])
    directory = file.with_suffix('')
    directory.mkdir(parents=True, exist_ok=True)
    # restart calculation from gpw file
    # don't write a new txt file
    atoms, calc = restart(str(file), txt=None)
    # generate cube files if orbitals
    # orbital files written: from HOMO+i to LUMO+a
    # change i and a according to your system
    gen_cubes(atoms, calc, str(Path(directory, file.stem)),
              i=i, a=a, spins=spins)
