from chem_utils import *

def test_frequency():
    fr=Frequency.load('tests/orca.out')
    assert(abs(fr.frequencies[0]-414.28) < 0.001)
    assert(fr.is_minimum)