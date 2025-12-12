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


def _mutate(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, mr: float) -> np.ndarray:
    """Gaussian mutation, clipped to [lb, ub]."""
    if mr <= 0:
        return x

    dim = x.size
    mask = np.random.rand(dim) < mr
    if not mask.any():
        return x

    # Step size scaled to range of each variable
    sigma = 0.05 * (ub - lb + 1e-12)
    noise = np.random.randn(dim) * sigma
    x_new = x.copy()
    x_new[mask] += noise[mask]
    return np.clip(x_new, lb, ub)


def run_ga(
        udp,
        pop_size: int = 80,
        num_generations: int = 400,
        tournament_size: int = 4,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.05,
        seed: int | None = None,
        log_interval: int = 25,
) -> Tuple[np.ndarray, float]:
    """
    Plain Genetic Algorithm that works directly with programmable_cubes_UDP.

    - Minimizes udp.fitness(x)  (smaller = better; works even if values are negative)
    - Uses:
        * tournament selection
        * uniform crossover
        * Gaussian mutation (clipped to bounds)

    Returns:
        best_x, best_fitness
    """
    start_time = time.time()

    lb, ub = udp.get_bounds()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    pop, fitness = _init_population(udp, pop_size, seed)
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]



    for gen in range(1, num_generations + 1):
        new_pop: List[np.ndarray] = []

        # Elitism: keep the current best
        new_pop.append(best_x.copy())

        # Fill the rest of the population
        while len(new_pop) < pop_size:
            p1 = _tournament_select(pop, fitness, tournament_size)
            p2 = _tournament_select(pop, fitness, tournament_size)

            c1, c2 = _crossover(p1, p2, crossover_rate)
            c1 = _mutate(c1, lb, ub, mutation_rate)
            c2 = _mutate(c2, lb, ub, mutation_rate)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        # Evaluate
        pop = new_pop
        fitness = [_evaluate(udp, x) for x in pop]

        # Track best
        current_best_idx = int(np.argmin(fitness))
        current_best_x = pop[current_best_idx]
        current_best_f = fitness[current_best_idx]

        if current_best_f < best_f:
            best_f = current_best_f
            best_x = current_best_x.copy()

        # Optional logging
        if log_interval and gen % log_interval == 0:
            avg_f = float(np.mean(fitness))
            print(
                f"[GA] Gen {gen:4d} | best = {best_f:.6f} | avg = {avg_f:.6f} "
                f"| elapsed = {time.time() - start_time:.1f}s"
            )

    print(f"[GA] Finished: best fitness = {best_f:.6f} | total time = {time.time() - start_time:.1f}s")
    return best_x, best_f
