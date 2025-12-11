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


def _init_population(
    udp,
    pop_size: int,
    seed: int | None = None
) -> Tuple[List[np.ndarray], List[float]]:
    """Uniform random initialization inside bounds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    lb, ub = udp.get_bounds()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.shape[0]

    pop: List[np.ndarray] = []
    fitness: List[float] = []

    for _ in range(pop_size):
        x = lb + (ub - lb) * np.random.rand(dim)
        f = _evaluate(udp, x)
        pop.append(x)
        fitness.append(f)

    return pop, fitness


def _tournament_select(pop: List[np.ndarray], fitness: List[float], k: int) -> np.ndarray:
    """Return index of best individual among k random picks (minimization)."""
    n = len(pop)
    indices = [random.randrange(n) for _ in range(k)]
    best_idx = min(indices, key=lambda i: fitness[i])
    return pop[best_idx].copy()

def _crossover(p1: np.ndarray, p2: np.ndarray, cr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple uniform crossover."""
    if random.random() > cr:
        return p1.copy(), p2.copy()

    mask = np.random.rand(p1.size) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


