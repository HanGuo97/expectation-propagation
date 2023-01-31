import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Optional, Callable

# Legacy reasons
from ..utils.misc_utils import dump, load


def get_damping_fn(damping_name: Optional[Union[str, float, Callable[[int], float]]], K: int) -> Callable[[int], float]:
    if callable(damping_name):
        return damping_name

    if isinstance(damping_name, str):
        if damping_name not in ["1/K", "decaying"]:
            raise ValueError

    if K < 2:
        raise ValueError

    def _damping_fn(t: int) -> float:
        if damping_name is None:
            return 1.0
        
        if isinstance(damping_name, float):
            return damping_name

        if damping_name == "1/K":
            return 1. / K

        if damping_name == "decaying":
            values = np.linspace(0.5, 1. / K, num=K).tolist()
            return values[min(K - 1, t)]

        raise ValueError

    return _damping_fn


def running_mean(values: List[float], window: int) -> List[float]:
    return pd.Series(values).rolling(window).mean().tolist()
