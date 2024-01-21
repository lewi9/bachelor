"""Module for custom types."""
from typing import Callable, Literal

import pandas as pd

# Modes
DownloadMode = Literal[None, "overwrite"]
ExtractionMode = Literal[None, "append", "overwrite"]
ForecastingPipelineMode = Literal[None, "recreate"]

# Modelling
ExoFiller = Literal["mean"]
EvaluatorMetric = Callable[[pd.Series, pd.Series], float]
