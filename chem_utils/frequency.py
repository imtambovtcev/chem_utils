import cclib
import numpy as np


class Frequency:
    def __init__(self, frquenceies, modes):
        self.frquenceies = frquenceies
        self.modes = modes

    @classmethod
    def load(cls, filename):
        data = cclib.io.ccread(filename)
        return cls(data.vibfreqs, data.vibdisps)

    @property
    def is_minimum(self):
        return np.all(self.frequencies > 0)

