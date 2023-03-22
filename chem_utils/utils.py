import numpy as np
import re
import json
import warnings
import pandas as pd

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Data(pd.DataFrame):
    @property
    def _constructor(self):
        return Data

    # def latex(self):
    #     from tabulate import tabulate
    #     return tabulate(self.__data__, headers='keys', tablefmt='latex')
#
#     @staticmethod
#     def load(filename):
#         with open(filename, 'r') as fp:
#             return Data(json.load(fp))
#
#     def plot(self, xkey, ykey, fig=None, ax=None, settings={}):
#         import matplotlib.pyplot as plt
#         if ax is None and fig is None:
#             fig, ax = plt.subplots()
#         ax.plot(self.__data__[xkey],self.__data__[ykey], **settings)
#         return fig, ax


def gaussian(x, b, c):  # no amplitude
    return np.exp(-(x - b) ** 2 / (2 * (c ** 2)))


class Spectrum:
    ch = 1239.87317526  # eV to nm coeff

    def __init__(self, energy, intensity):
        """
        :param energy: energy for the spectrum in eV
        :param intensity: intensity in any units
        """
        self.energy = np.array(energy)
        self.intensity = np.array(intensity)
        assert len(self.energy) == len(self.intensity)

    def get_spectrum(self, wavelength_limits=None, N_points=200, sigma=0.1, fwhm=None, normalize=False):
        """
        returns wavelength np.array in nm and intensity in utits from init
        :param sigma: width in eV (works if fwhm undefined)
        :param fwhm: Full width at half maximum
        """
        if fwhm is not None:
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        if wavelength_limits is None:
            wl_min = self.ch / self.energy.max()
            wl_max = self.ch / self.energy.min()
        else:
            wl_min = np.min(wavelength_limits)
            wl_max = np.max(wavelength_limits)

        wl_arr = np.linspace(wl_min, wl_max, N_points)
        energy_arr = self.ch / wl_arr

        x, b = np.meshgrid(energy_arr, self.energy)

        intensity = np.dot(self.intensity, gaussian(
            x, b, sigma)) / (sigma * np.sqrt(2 * np.pi))
        if normalize:
            intensity /= intensity.max()

        return wl_arr, intensity

    def plot_spectrum(self, fig, ax, wavelength_limits=None, N_points=200, sigma=0.1, fwhm=None, normalize=False,
                      label=None):
        ax.plot(*self.get_spectrum(wavelength_limits=wavelength_limits, N_points=N_points, sigma=sigma, fwhm=fwhm,
                                   normalize=normalize), label=label)

    def plot_spectrum_peaks(self, fig, ax, wavelength_limits=None, normalize=False, color='r', label=None):
        if wavelength_limits is None:
            wl_min = self.ch / self.energy.max()
            wl_max = self.ch / self.energy.min()
        else:
            wl_min = np.min(wavelength_limits)
            wl_max = np.max(wavelength_limits)
        wl = self.ch / self.energy
        ind = np.logical_and(wl > wl_min, wl < wl_max)
        wl = wl[ind]
        intensity = self.intensity[ind]
        if normalize:
            intensity /= intensity.max()
        put_label_flag = True
        for w, i in zip(wl, intensity):
            if put_label_flag:
                ax.plot([w, w], [0, i], c=color, label=label)
                put_label_flag = False
            else:
                ax.plot([w, w], [0, i], c=color)

    def energy_peak_by_wavelength(self, wl):
        return self.energy[np.argmin(np.abs(self.energy - self.ch / wl))]

    def highest_peak_in_range(self, wavelength_min=None, wavelength_max=None):
        # print(self)
        _wl = self.ch/self.energy
        _intensity = np.copy(self.intensity)
        if wavelength_min is not None:
            _intensity = _intensity[_wl >= wavelength_min]
            _wl = _wl[_wl >= wavelength_min]
        if wavelength_max is not None:
            _intensity = _intensity[_wl <= wavelength_max]
            _wl = _wl[_wl <= wavelength_max]
        # print(f'{_wl = }')
        # print(f'{_intensity = }')
        return _wl[np.argmax(_intensity)]

    def __str__(self):
        from pandas import DataFrame
        return str(DataFrame({'Energy, eV': self.energy, 'wavelength, nm': self.ch / self.energy, 'intensity': self.intensity}))


def get_last_occurrence(filename, bstring):
    with open(filename) as f:
        lines = f.readlines()
        for line in lines[::-1]:
            if line.startswith(bstring):
                return line
    return ''


#     parse the time string to give back the time as a float:
def parse_time(string):
    # print(f'{string = }')
    try:
        r = rx.findall(string)
        time = float(r[0])
    except Exception as e:
        # print(Exception)
        time = 0.0
    return time


def energy_from_ocra(filename):
    """
    Energy in Eh
    """
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines[::-1]):
            if line.startswith('FINAL SINGLE POINT ENERGY'):
                break

        d = rx.findall(line)
        return float(d[0])


def energy_from_ocra_ev(filename):
    """
    Energy in eV
    """
    return 27.2113961318065*energy_from_ocra(filename)


def surface_energy_from_ocra(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines[::-1]):
            if line.startswith('The Calculated Surface using the'):
                break
        # print(f'{id = }')
        # print(f'{line = }')
        id = len(lines) - id
        for line in lines[id + 1:]:
            if line.startswith('------------------') or len(line) <= 2:
                break
            d = rx.findall(line)
            try:
                data = pd.concat([data, pd.DataFrame(
                    {'param': [float(d[0])], 'E(Eh)': [float(d[1])]})], ignore_index=True)
            except:
                pass

    return data


def energy_from_gpaw(filename):
    try:
        with open(filename) as f:
            lines = f.readlines()
            for id, line in enumerate(lines):
                if line.startswith('Extrapolated'):
                    break
            try:
                d = rx.findall(line)
                energy = float(d[0])
            except:
                energy = np.nan
            return energy
    except:
        return np.nan


def orbital_energy_from_ocra(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines[::-1]):
            if line.startswith('ORBITAL ENERGIES'):
                break
        # print(f'{id = }')
        # print(f'{line = }')

        for line in lines[id + 4:]:
            if line.startswith('------------------') or len(line) <= 2:
                break
            d = rx.findall(line)
            data = pd.concat([data, pd.DataFrame({'NO': [float(d[0])], 'OCC': [float(
                d[1])], 'E(Eh)': [float(d[2])], 'E(eV)': [float(d[3])]})], ignore_index=True)

    return data


def spectrum_from_gpaw_linear(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            if line.startswith('#       energy'):
                break
        # print(f'{id = }')
        # print(f'{line = }')

        for line in lines[id + 1:]:
            # print(line)
            # print(len(line))
            if line.startswith('------------------') or len(line) <= 2:
                break
            d = rx.findall(line)
            # print(f'{d = }')
            data = pd.concat([data, pd.DataFrame(
                {'energy': [float(d[0])], 'osc str': [float(d[1])], 'rot str': [float(d[2])], 'osc str x': [float(d[3])],
                 'osc str y': [float(d[4])], 'osc str z': [float(d[5])]})], ignore_index=True)

    return data


def orbital_energy_from_gpaw(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines[::-1]):
            if line.startswith(' Band  Eigenvalues  Occupancy'):
                break
        try:
            id = len(lines) - id
            # print(f'{id = }')
            # print(f'{line = }')

            for line in lines[id + 1:]:
                # print(line)
                # print(len(line))
                if line.startswith('------------------') or len(line) <= 2:
                    break
                d = rx.findall(line)
                # print(f'{d = }')
                try:
                    data = pd.concat([data, pd.DataFrame({'Band': [float(d[0])], 'Eigenvalues': [
                                     float(d[1])], 'Occupancy': [float(d[2])]})], ignore_index=True)
                except (IndexError, TypeError):
                    warnings.warn(
                        'File {} has no orbitals info'.format(filename))
        except:
            warnings.warn('File {} load fail'.format(filename))

    return data


def orbital_energy_from_gpaw_up_down(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines[::-1]):
            if line.startswith(' Band  Eigenvalues  Occupancy'):
                break

        try:
            id = len(lines) - id
            # print(f'{id = }')
            # print(f'{line = }')

            for line in lines[id + 1:]:
                # print(line)
                # print(len(line))
                if line.startswith('------------------') or len(line) <= 2:
                    break
                d = rx.findall(line)
                # print(f'{d = }')
                try:
                    data = pd.concat([data, pd.DataFrame({'Band': [float(d[0])], 'Eigenvalues up': [float(d[1])], 'Occupancy  up': [float(d[2])],
                                                         'Eigenvalues down': [float(d[3])], 'Occupancy  down': [float(d[4])]})], ignore_index=True)
                except (IndexError, TypeError):
                    warnings.warn(
                        'File {} has no orbitals info'.format(filename))
        except:
            warnings.warn('File {} load fail'.format(filename))

    return data


def spectrum_from_orca_e(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            if line.startswith('         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'):
                break
        # print(f'{id = }')
        # print(f'{line = }')

        for line in lines[id + 5:]:
            # print(line)
            # print(len(line))
            if line.startswith('------------------') or len(line) <= 2:
                break
            d = rx.findall(line)
            # print(f'{d = }')
            data = pd.concat([data, pd.DataFrame({'State': [float(d[0])], 'Energy(cm-1)': [float(d[1])], 'Wavelength(nm)': [float(d[2])], 'fosc': [float(d[3])],
                                                 'T2(au**2)': [float(d[4])], 'TX(au)': [float(d[5])], 'TY(au)': [float(d[6])], 'TZ(au)': [float(d[7])]})], ignore_index=True)

    return data


def spectrum_from_orca_v(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            if line.startswith('         ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'):
                break
        # print(f'{id = }')
        # print(f'{line = }')

        for line in lines[id + 5:]:
            # print(line)
            # print(len(line))
            if line.startswith('------------------') or len(line) <= 2:
                break
            d = rx.findall(line)
            # print(f'{d = }')
            data = pd.concat([data, pd.DataFrame({'State': [float(d[0])], 'Energy(cm-1)': [float(d[1])], 'Wavelength(nm)': [float(d[2])], 'fosc': [float(d[3])],
                                                 'T2(au**2)': [float(d[4])], 'TX(au)': [float(d[5])], 'TY(au)': [float(d[6])], 'TZ(au)': [float(d[7])]})], ignore_index=True)

    return data


def spectrum_from_gpaw(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            if line.startswith('#    om (eV)'):
                break
        # print(f'{id = }')
        # print(f'{line = }')

        for line in lines[id + 1:]:
            # print(line)
            # print(len(line))
            if line.startswith('------------------') or len(line) <= 2:
                break
            d = rx.findall(line)
            # print(f'{d = }')
            data = pd.concat([data, pd.DataFrame({'om (eV)': [float(d[0])], 'S_x': [float(
                d[1])], 'S_y': [float(d[2])], 'S_z': [float(d[3])]})], ignore_index=True)

    return data


def neb_energy_from_ocra(filename):
    data = pd.DataFrame()
    with open(filename) as f:
        lines = f.readlines()
        file_key = 'error'
        for id, line in enumerate(lines[::-1]):
            if 'PATH SUMMARY' in line:
                file_key = 'CI'
                break
            if line.startswith('                      PATH SUMMARY'):
                file_key = 'TS'
                break
        id = len(lines) - id
        # print(f'{id = }')
        # print(f'{line = }')

        for i, line in enumerate(lines[id + 4:]):
            # print(f'{line = }')
            if line.startswith('--------') or len(line) <= 2:
                break
            if file_key == 'TS':
                d = rx.findall(line)
                if 'TS' in line:
                    data = pd.concat([data, pd.DataFrame({'Image': [i], 'E(Eh)': [float(d[0])], 'dE(kcal/mol)': [
                        float(d[1])], 'max(|Fp|)': [float(d[2])], 'RMS(Fp)': [float(d[3])], 'Comment': ['TS']})],
                        ignore_index=True)
                elif 'CI' in line:
                    data = pd.concat([data, pd.DataFrame({'Image': [i], 'E(Eh)': [float(d[1])], 'dE(kcal/mol)': [
                        float(d[2])], 'max(|Fp|)': [float(d[3])], 'RMS(Fp)': [float(d[4])], 'Comment': ['CI']})],
                        ignore_index=True)
                else:
                    data = pd.concat([data, pd.DataFrame({'Image': [i], 'E(Eh)': [float(d[1])], 'dE(kcal/mol)': [
                        float(d[2])], 'max(|Fp|)': [float(d[3])], 'RMS(Fp)': [float(d[4])], 'Comment': ['']})],
                        ignore_index=True)
            if file_key == 'CI':
                d = rx.findall(line)
                if 'CI' in line:
                    data = pd.concat([data, pd.DataFrame(
                        {'Image': [i], 'Dist.(Ang.)': [float(d[1])], 'E(Eh)': [float(d[2])], 'dE(kcal/mol)': [
                            float(d[3])], 'max(|Fp|)': [float(d[4])], 'RMS(Fp)': [float(d[5])], 'Comment': ['CI']})],
                        ignore_index=True)
                else:
                    data = pd.concat([data, pd.DataFrame(
                        {'Image': [i], 'Dist.(Ang.)': [float(d[1])], 'E(Eh)': [float(d[2])], 'dE(kcal/mol)': [
                            float(d[3])], 'max(|Fp|)': [float(d[4])], 'RMS(Fp)': [float(d[5])], 'Comment': ['']})],
                        ignore_index=True)

    data['Energy, eV'] = 27.2114 * (data['E(Eh)'] - data['E(Eh)'].iloc[0])

    return data


def neb_energy_from_ocra(filename):
    data = pd.DataFrame()
    try:
        with open(filename) as f:
            lines = f.readlines()
            file_key = 'error'
            for id, line in enumerate(lines[::-1]):
                if 'PATH SUMMARY FOR NEB-TS' in line:
                    file_key = 'TS'
                    break
                if 'PATH SUMMARY' in line:
                    file_key = 'CI'
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
                    if 'TS' in line:
                        data = pd.concat([data, pd.DataFrame({'Image': [i], 'E(Eh)': [float(d[0])], 'dE(kcal/mol)': [
                            float(d[1])], 'max(|Fp|)': [float(d[2])], 'RMS(Fp)': [float(d[3])], 'Comment': ['TS']})], ignore_index=True)
                    elif 'CI' in line:
                        data = pd.concat([data, pd.DataFrame({'Image': [i], 'E(Eh)': [float(d[1])], 'dE(kcal/mol)': [
                            float(d[2])], 'max(|Fp|)': [float(d[3])], 'RMS(Fp)': [float(d[4])], 'Comment': ['CI']})], ignore_index=True)
                    else:
                        data = pd.concat([data, pd.DataFrame({'Image': [i], 'E(Eh)': [float(d[1])], 'dE(kcal/mol)': [
                            float(d[2])], 'max(|Fp|)': [float(d[3])], 'RMS(Fp)': [float(d[4])], 'Comment': ['']})], ignore_index=True)
                if file_key == 'CI':
                    d = rx.findall(line)
                    if 'CI' in line:
                        data = pd.concat([data, pd.DataFrame({'Image': [i], 'Dist.(Ang.)': [float(d[1])], 'E(Eh)': [float(d[2])], 'dE(kcal/mol)': [
                            float(d[3])], 'max(|Fp|)': [float(d[4])], 'RMS(Fp)': [float(d[5])], 'Comment': ['CI']})], ignore_index=True)
                    else:
                        data = pd.concat([data, pd.DataFrame({'Image': [i], 'Dist.(Ang.)': [float(d[1])], 'E(Eh)': [float(d[2])], 'dE(kcal/mol)': [
                            float(d[3])], 'max(|Fp|)': [float(d[4])], 'RMS(Fp)': [float(d[5])], 'Comment': ['']})], ignore_index=True)

    except Exception as e:
        print(e)
        data = pd.DataFrame({'Image': [0], 'Dist.(Ang.)': [0.], 'E(Eh)': [np.nan], 'dE(kcal/mol)': [
            np.nan], 'max(|Fp|)': [np.nan], 'RMS(Fp)': [np.nan], 'Comment': ['']})
    data['Energy, eV'] = 27.2114*(data['E(Eh)'])
    return data


def neb_energy_from_ocra_interp(filename):
    points = pd.DataFrame()
    interpolation = pd.DataFrame()
    try:
        with open(filename) as f:
            lines = f.readlines()
            for id, line in enumerate(lines):
                if 'Images' in line:
                    break
            # print(f'{id = }')
            # print(f'{line = }')

            for i, line in enumerate(lines[id + 1:]):
                # print(f'{line = }')
                if line.startswith('--------') or len(line) <= 2:
                    break

                d = rx.findall(line)
                points = pd.concat([points, pd.DataFrame({'Image': [i], 'Images': [float(d[0])], 'Distance (Bohr)': [
                    float(d[1])], 'Energy (Eh)': [float(d[2])]})], ignore_index=True)

            # print(f'{i = }')
            # print(f'{line = }')

            for j, line in enumerate(lines[id + i + 4:]):
                # print(f'{line = }')
                if line.startswith('--------') or len(line) <= 2:
                    break

                d = rx.findall(line)
                interpolation = pd.concat([interpolation, pd.DataFrame({'Interp': [float(d[0])], 'Distance (Bohr)': [
                    float(d[1])], 'Energy (Eh)': [float(d[2])]})], ignore_index=True)
    except Exception as e:
        print(e)
        points = pd.DataFrame({'Image': [0], 'Images': [0.], 'Distance (Bohr)': [
            0.], 'Energy (Eh)': [np.nan]})
        interpolation = pd.DataFrame({'Interp': [0.], 'Distance (Bohr)': [
            0.], 'Energy (Eh)': [np.nan]})

    points['Energy, eV'] = 27.2114 * \
        (points['Energy (Eh)']-points['Energy (Eh)'].iloc[0])
    interpolation['Energy, eV'] = 27.2114 * \
        (interpolation['Energy (Eh)']-interpolation['Energy (Eh)'].iloc[0])

    return points, interpolation


def neb_free_energy_from_ocra(filename):
    try:
        with open(filename) as f:
            lines = f.readlines()
            free_energy = np.nan
            for id, line in enumerate(lines[::-1]):
                if 'Final Gibbs free energy' in line:
                    d = rx.findall(line)
                    free_energy = float(d[-1])
                    break
    except Exception as e:
        print(e)
        return np.nan

    return 27.2114*free_energy


def orca_frequencies(filename):
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
        print(e)
        data = pd.DataFrame({'Frequency, cm**-1': [np.nan], 'Comment': ['']})
    return data
