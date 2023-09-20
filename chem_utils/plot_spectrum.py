import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .spectrum import Spectrum
import argparse


def plot_spectrum(file_list, vlim = [300, 500], sigma = 0.1, savefig='spectrum.pdf', normalize = True):
    print(f'{normalize = }')
    fig, ax = plt.subplots()
    for file in file_list:
        spectrum = Spectrum.load_orca(file)
        spectrum.plot_spectrum(fig, ax, wavelength_limits=vlim, sigma=sigma, normalize=normalize, label=str(file.parent))
    plt.xlabel(r'$\lambda, nm$')
    plt.ylabel(r'Intensity')
    plt.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('inputs', metavar='I', type=str, nargs='*', default=['./'],
                        help='an input directory or file for processing')

    # Adding plot_spectrum arguments to the parser
    parser.add_argument('--vlim', type=int, nargs=2, default=[300, 500],
                        help='wavelength limits for the spectrum plot')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='sigma for the spectrum plot')
    parser.add_argument('--savefig', type=str, default='spectrum.pdf',
                        help='name of the output figure file')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True,
                    help='whether to normalize the spectrum')


    args = parser.parse_args()

    _input = args.inputs
    print(_input)
    _input = [Path(i) for i in _input]
    inp = []
    for i in _input:
        if i.is_dir():
            add = i.glob('*.xyz')
            inp.extend(add)
        elif i.is_file():
            inp.append(i)
    print(inp)

    # Passing the parsed arguments to plot_spectrum
    plot_spectrum(inp, vlim=args.vlim, sigma=args.sigma, savefig=args.savefig, normalize=args.normalize)


if __name__ == "__main__":
    main()
