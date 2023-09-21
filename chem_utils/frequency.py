import re
import warnings

import numpy as np
import pandas as pd

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


class Frequency:
    def __init__(self, df):
        self.data = df

    @classmethod
    def load(cls, filename):
        data = pd.DataFrame()
        try:
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
        except Exception as e:
            warnings.warn(e)
            data = pd.DataFrame(
                {'Frequency, cm**-1': [np.nan], 'Comment': ['']})
        return cls(data)

    def check_6_modes_are_zero(self):
        assert np.all(abs(self.data['Frequency, cm**-1'])[:6] < 1e-6)

    @property
    def all_frequencies(self):
        return self.data['Frequency, cm**-1'].values

    @property
    def nonzero_frequencies(self):
        self.check_6_modes_are_zero()
        return self.data['Frequency, cm**-1'].values[6:]

    @property
    def is_minimum(self):
        return np.all(self.nonzero_frequencies > 0)
