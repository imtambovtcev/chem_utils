import numpy as np
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from .utils import neb_energy_from_ocra

import re
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


def neb_energy_from_ocra_interp(filename):
    points = pd.DataFrame()
    interpolation = pd.DataFrame()
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

    points['Energy, eV'] = 27.2114 * \
        (points['Energy (Eh)']-points['Energy (Eh)'].iloc[0])
    interpolation['Energy, eV'] = 27.2114 * \
        (interpolation['Energy (Eh)']-interpolation['Energy (Eh)'].iloc[0])

    return points, interpolation


def neb_plot(file, show=True, save=None, title=None, sign_barriers=True, fig=None, ax=None, label=None, mode='first'):
    if isinstance(file, tuple) or isinstance(file, list):
        file_interpolate, file_out = file
    else:
        file_interpolate = file
        file_out = None
    try:
        points, interpolation = neb_energy_from_ocra_interp(file_interpolate)
        # print('Barrier = {}'.format(
        #     points['Energy, eV'].max()-points['Energy, eV'].iloc[-1]))
        # print('Barrier = {}'.format(
        #     points['Energy, eV'].max()-points['Energy, eV'].iloc[0]))

        if file_out is None:
            x_ts, y_ts = points['Images'].iloc[points['Energy, eV'].argmax(
            )], points['Energy, eV'].max()
        else:
            data = neb_energy_from_ocra(str(file_out))
            # print(data)
            if 'TS' in np.array(data['Comment']):
                x_ts = interpolation['Interp'].iloc[interpolation['Energy, eV'].argmax(
                )]
                y_ci = interpolation['Energy, eV'].max()
                # print(np.array(data['Comment']) == 'TS')
                y_ts = np.array(data['Energy, eV'])[np.array(
                    data['Comment']) == 'TS'][0]-data['Energy, eV'].iloc[0]
                # print(x_ts, y_ci, x_ts, y_ts)
                ax.arrow(x_ts, y_ci, 0, y_ts-y_ci, head_width=0.01,
                         head_length=0.03, fc='r', ec='r', length_includes_head=True)
            else:
                x_ts, y_ts = points['Images'].iloc[points['Energy, eV'].argmax(
                )], points['Energy, eV'].max()
        # print(f'{file_interpolate = } {file_out = }')

        match mode:
            case 'saddle':
                energy_ofset = points['Energy, eV'].iloc[0]+y_ts
            case 'last':
                energy_ofset = points['Energy, eV'].iloc[-1]
            case _:
                energy_ofset = points['Energy, eV'].iloc[0]

        if fig is None:
            fig, ax = plt.subplots()
        line1, = ax.plot(interpolation['Interp'],
                         interpolation['Energy, eV']-energy_ofset, label=label)
        line_color = line1.get_color()
        ax.plot(points['Images'], points['Energy, eV']-energy_ofset, '.', color=line_color)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Energy, eV')

        if label is not None:
            plt.legend()
        # interpolation.plot(ax=ax, x='Interp', y='Energy, eV')
        # points.plot.scatter(ax=ax, x='Images', y='Energy, eV')

        x_a, y_a = 0., points['Energy, eV'].iloc[0]-energy_ofset
        x_b, y_b = 1., points['Energy, eV'].iloc[-1]-energy_ofset

        if sign_barriers:
            ax.plot([x_ts, x_a], [y_ts, y_ts], 'k')
            ax.plot([x_ts, x_b], [y_ts, y_ts], 'k')
            ax.arrow(x_a, y_a, 0, y_ts-y_a, head_width=0.01, head_length=0.03, fc='k',
                     ec='k', length_includes_head=True)
            ax.arrow(x_b, y_b, 0, y_ts-y_b, head_width=0.01, head_length=0.03, fc='k',
                     ec='k', length_includes_head=True)
            plt.text(x_a+0.02, (y_a+y_ts)/2, "{:.2f} eV".format(y_ts-y_a), size=10, rotation=90.,
                     ha="left", va="center"
                     )
            plt.text(x_b-0.02, (y_b+y_ts)/2, "{:.2f} eV".format(y_ts-y_b), size=10, rotation=90.,
                     ha="right", va="center"
                     )
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        if save:
            print('Saving... ')
            if isinstance(save, str):
                plt.savefig(save)
            else:
                plt.savefig(file_interpolate.parent /
                            (file_interpolate.stem + '.pdf'))
        if show:
            plt.show()
        return fig, ax
    except Exception as e:
        print(e)
        fig, ax = plt.subplots()
        return fig, ax


def main():
    _input = ['./'] if len(sys.argv) <= 1 else sys.argv[1:]
    print(_input)
    _input = [Path(d) for d in _input]
    input = []
    for d in _input:
        if d.is_dir():
            if (d/'orca.out').is_file():
                add_out = d/'orca.out'
                input.append((d/'orca.final.interp', add_out))
            else:
                add_interpolate = d.glob('*.final.interp')
                input.extend(add_interpolate)
        else:
            input.append(d)
    print(input)
    [neb_plot(file, save=True, show=True) for file in input]


if __name__ == "__main__":
    main()
