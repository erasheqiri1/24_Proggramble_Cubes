import sys
import time

from core.programmable_cubes_UDP import programmable_cubes_UDP
from algorithm.ga_solver import run_ga


VALID_PROBLEMS = ("ISS", "JWST", "Enterprise")


def parse_problem(argv) -> str | None:
    # Lexon problemin nga command line (-p/--problem) dhe kthen emrin ose None nëse nuk jepet
    for i, arg in enumerate(argv):
        if arg in ("--problem", "-p") and i + 1 < len(argv):
            return argv[i + 1]
    return None


def run_problem(problem_name: str):

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

    # Krijo UDP për instancen e zgjedhur
    udp = programmable_cubes_UDP(problem_name)

    start_time = time.time()

    best_x, best_f = run_ga(
        udp,
        num_generations=400,
        log_interval=5
    )

    elapsed = time.time() - start_time

    print("-" * 70)
    print(f"[{problem_name}] Best fitness achieved: {best_f:.6f}")
    print(f"[{problem_name}] Total time: {elapsed:.1f} s")
    print("-" * 70)

    return best_x, best_f
if __name__ == "__main__":
    # Lexo problem nga komand line nese jepet
    problem = parse_problem(sys.argv)

    if problem is None:
        # Default nese s'jepet asgje
        print("No --problem specified. Using default: ISS")
        problem = "ISS"

    best_solution, best_fitness = run_problem(problem)

    print("\nDone.")
