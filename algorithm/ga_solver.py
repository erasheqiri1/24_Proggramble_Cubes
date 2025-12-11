import time
import random
from typing import Tuple, List

import numpy as np

def _evaluate(udp, x: np.ndarray) -> float:
    """Evaluate decision vector x with the UDP, return scalar fitness."""
    # udp.fitness usually returns a 1-element sequence
    val = udp.fitness(x)
    if isinstance(val, (list, tuple, np.ndarray)):
        return float(val[0])
    return float(val)