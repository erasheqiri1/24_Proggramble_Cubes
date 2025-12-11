import sys
import time

from core.programmable_cubes_UDP import programmable_cubes_UDP
from algorithm.ga_solver import run_ga


VALID_PROBLEMS = ("ISS", "JWST", "Enterprise")


def parse_problem(argv) -> str | None:
    """
    Lexon nga komand line:
        python main.py --problem ISS
        python main.py -p JWST
    Kthen emrin e problemit ose None nëse s'është dhënë.
    """
    for i, arg in enumerate(argv):
        if arg in ("--problem", "-p") and i + 1 < len(argv):
            return argv[i + 1]
    return None

    # Krijo UDP për instancën e zgjedhur
    udp = programmable_cubes_UDP(problem_name)

    start_time = time.time()


def run_problem(problem_name: str):
    """
    Run ONLY GA for problem_name.
    """
    if problem_name not in VALID_PROBLEMS:
        raise ValueError(
            f"Unknown problem '{problem_name}'. "
            f"Valid options: {', '.join(VALID_PROBLEMS)}"
        )

    print("\n" + "=" * 70)
    print(f"Running problem: {problem_name}")
    print("=" * 70)
    print("Algorithm: Genetic Algorithm (GA)")
    print("-" * 70)

    # Krijo UDP për instancën e zgjedhur
    udp = programmable_cubes_UDP(problem_name)

    start_time = time.time()
