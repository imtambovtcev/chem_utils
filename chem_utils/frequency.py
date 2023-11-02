import cclib
import numpy as np
import re
import pandas as pd


class Frequency:
    def __init__(self, frequencies, modes):
        self.frequencies = frequencies
        self.modes = modes

    @classmethod
    def load(cls, filename):
        try:
            data = cclib.io.ccread(filename)
            vibfreqs = data.vibfreqs
            vibdisps = data.vibdisps
        except:
            numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
            rx = re.compile(numeric_const_pattern, re.VERBOSE)
            data = pd.DataFrame()
            with open(filename) as f:
                lines = f.readlines()
                file_key = 'error'
                for id, line in enumerate(lines[::-1]):
                    if 'VIBRATIONAL FREQUENCIES' in line:
                        file_key = 'TS'
                        break
                id = len(lines) - id
                # print(f'{id = }')
                # print(f'{line = }')
                # print(f'{file_key = }')

                for i, line in enumerate(lines[id + 4:]):
                    # print(f'{line = }')
                    if line.startswith('--------') or len(line) <= 2:
                        break
                    if file_key == 'TS':
                        d = rx.findall(line)
                        if 'imaginary mode' in line:
                            comment = 'imaginary mode'
                        else:
                            comment = ''
                        data = pd.concat([data, pd.DataFrame(
                            {'Frequency, cm**-1': [float(d[1])], 'Comment': [comment]})], ignore_index=True)
                # cclib failed
            data = data[data['Frequency, cm**-1'] != 0]
            vibfreqs = data['Frequency, cm**-1']
            vibdisps = None

        return cls(vibfreqs, vibdisps)

    @property
    def is_minimum(self):
        return np.all(self.frequencies > 0)
