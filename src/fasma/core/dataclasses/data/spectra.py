from dataclasses import dataclass
from abc import ABC
import pandas as pd


class SpectraData(ABC):
    "Spectra Data"


@dataclass
class RealTimeData(SpectraData):
    n_step: int
    steps: pd.DataFrame
