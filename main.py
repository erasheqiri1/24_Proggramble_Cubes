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