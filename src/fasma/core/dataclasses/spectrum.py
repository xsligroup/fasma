from fasma.core import functional
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Spectrum:
    freq: np.ndarray
    spect: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def data_to_hdf5(self, file, name):
        group = file.create_group(name)
        group.create_dataset("freq", data=self.freq)
        group.create_dataset("spect", data=self.spect)
        group.create_dataset("x", data=self.x)
        group.create_dataset("y", data=self.y)
        group.attrs['type'] = "Spectrum"


@dataclass
class RTimeSpectrum(Spectrum):
    def gen_spect(self, damp: float = 0.0001, wlim: tuple = (0, 4/27), res: float = 400000, every_step: int = 100, meth: str = "pade"):
        meth = meth.lower()
        chosen_times = self.x[::every_step]
        chosen_dipoles = self.x[::every_step]
        step_size = chosen_times[1] - chosen_times[0]
        spects = []
        for x in range(3):
            s = chosen_dipoles[:, x]
            s -= s[0]
            s *= np.exp[-damp * chosen_times]
            if meth == "pade":
                transformer = functional.pade_tx
            elif meth == "gaussian":
                transformer = functional.fourier_tx
            else:
                raise ValueError('Unsupported distribution "{0}" specified'.format(meth))
            self.freq, f = transformer(s, step_size, wlim, res)
            spects.append(f)
        self.spect = sum([f.imag for f in spects]) / 3
        self.spect *= -self.freq


@dataclass
class ImportedSpectrum(Spectrum):
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    def data_to_hdf5(self, file, name):
        group = file.create_group(name)
        group.create_dataset("freq", data=self.freq)
        group.create_dataset("spect", data=self.spect)
        group.attrs['type'] = "Imported"


@dataclass
class SimulatedSpectrum(Spectrum):
    freq: np.ndarray = field(init=False)
    spect: np.ndarray = field(init=False)

    def add_broadened(self, freq, spect):
        self.freq = freq
        self.spect = spect

    def gen_spect(self, broad: float = 0.5, wlim=None, res: float = 100, xshift: float = 0, meth: str = 'lorentz'):
        meth = meth.lower()
        nani = False
        if meth == 'lorentz':
            if nani:
                meth = functional.lorentzian_2
            else:
                meth = functional.lorentzian
        elif meth == 'gaussian':
            if nani:
                meth = functional.gaussian_2
            else:
                meth = functional.gaussian
        else:
            raise ValueError('Unsupported distribution "{0}" specified'.format(meth))

        if wlim is None:
            print("Spectral range not specified... " +
                  "Automatically generating spectral range")
            percent = 0.930

            # Find max and min energies
            min = self.x.min()
            max = self.x.max()

            # Use quartile function of lorentz distribution regardless of distribution type
            lower_bound = broad * np.tan(((1 - percent) - 0.5) * np.pi) + min
            upper_bound = broad * np.tan((percent - 0.5) * np.pi) + max
            wlim = (lower_bound, upper_bound)
        n_points = int((wlim[1] - wlim[0]) * res)

        self.freq = np.linspace(wlim[0], wlim[1], n_points) + xshift
        self.spect = np.zeros(n_points)

        for current_x, current_y in zip(self.x, self.y):
            self.spect += meth(broad, current_x, current_y, self.freq)

    def data_to_hdf5(self, file, name):
        group = file.create_group(name)
        freq = getattr(self, 'freq', None)
        spect = getattr(self, 'spect', None)
        if freq is not None:
            group.create_dataset("freq", data=freq)
        if spect is not None:
            group.create_dataset("spect", data=spect)
        group.create_dataset("x", data=self.x)
        group.create_dataset("y", data=self.y)
        group.attrs['type'] = "Simulated"


def hdf5_to_data(group):
    spectrum_type = group.attrs['type']
    key_list = group.keys()
    if "x" in key_list:
        x = group["x"][:]
        y = group["y"][:]
    else:
        x = None
        y = None
    if "freq" in key_list:
        freq = group["freq"][:]
        spect = group["spect"][:]
    else:
        freq = None
        spect = None
    if spectrum_type == "Imported":
        spectrum = ImportedSpectrum(freq=freq, spect=spect)
    elif spectrum_type == "Simulated":
        spectrum = SimulatedSpectrum(x=x, y=y)
        spectrum.add_broadened(freq=freq, spect=spect)
    else:
        spectrum = Spectrum(freq=freq, spect=spect, x=x, y=y)
    return spectrum
