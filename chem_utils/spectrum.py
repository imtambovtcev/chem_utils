import numpy as np
import re
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def gaussian(x, b, c):  # no amplitude
    return np.exp(-(x - b) ** 2 / (2 * (c ** 2)))


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

    @classmethod
    def load_orca(cls, filename, mode='e'):
        if mode == 'v':
            data = spectrum_from_orca_v(filename)
        else:
            data = spectrum_from_orca_e(filename)
        if data.empty:
            raise ValueError("Empty DataFrame encountered!")
        return cls(data['Energy(cm-1)']/8065.54429, data['T2(au**2)'])

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

    def plot_spectrum(self, fig=None, ax=None, wavelength_limits=None, N_points=200, sigma=0.1, fwhm=None, normalize=False,
                      label=None):
        # Create new fig and ax if they are None
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(*self.get_spectrum(wavelength_limits=wavelength_limits, N_points=N_points, sigma=sigma, fwhm=fwhm,
                                   normalize=normalize), label=label)
        
        return fig, ax

    def plot_spectrum_peaks(self, fig=None, ax=None, wavelength_limits=None, normalize=False, color='r', label=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
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

        return fig, ax

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
