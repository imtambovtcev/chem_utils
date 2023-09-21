from chem_utils import *

def test_frequency():
    fr=Frequency.load('tests/orca.out')
    assert(fr.all_frequencies[0] == 0.0)
    assert(abs(fr.all_frequencies[6]-414.28) < 0.001)
    assert(abs(fr.nonzero_frequencies[0]-414.28) < 0.001)
    assert(fr.is_minimum)