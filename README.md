## Programmable Cubes – Genetic Algorithm Solution

This project implements a Genetic Algorithm (GA) to solve the Programmable Cubes optimization problem from the ESA Space Optimization Competition (SpOC3).
The problem involves rearranging modular cubes into target configurations under strict physical constraints, resulting in a large combinatorial search space. Due to its complexity, heuristic optimization methods are required.

### Problem Description

Three official ESA instances are considered:
* ISS – low complexity
* JWST – medium complexity
* Enterprise – high complexity

Each instance differs in search space size and constraint density.

### Methodology

A Genetic Algorithm was selected because exhaustive and deterministic search methods are infeasible for this problem.

Key characteristics:
* Population-based evolutionary search
* Tournament selection
* Crossover and mutation
* Elitism
* Early stopping based on stagnation
* Fitness evaluation performed exclusively via the ESA-provided UDP

The implementation is fully ESA-compliant and does not rely on custom heuristics or manual fitness adjustments.

## Project Structure
```
24_Programmable_Cubes/
│
├── algorithm/
│   └── ga_solver.py
│       # Implementation of the Genetic Algorithm:
│       # population initialization, selection, crossover, mutation,
│       # elitism, stopping criteria, and interaction with the UDP.
│
├── core/
│   ├── CubeMoveset.py
│   │   # Definition of valid cube moves and movement constraints
│   │   # according to the ESA Programmable Cubes specification.
│   │
│   └── programmable_cubes_UDP.py
│       # ESA-compliant User Defined Problem (UDP):
│       # loads problem instances, applies moves, and evaluates fitness
│       # using the official ESA rules.
│
├── data/
│   ├── Enterprise/
│   │   # Instance-specific data for the Enterprise scenario.
│   │
│   ├── ISS/
│   │   # Instance-specific data for the ISS scenario.
│   │
│   └── JWST/
│       # Instance-specific data for the JWST scenario.
│
├── problems/
│   ├── Enterprise.json
│   │   # Official ESA problem definition for the Enterprise instance.
│   │
│   ├── ISS.json
│   │   # Official ESA problem definition for the ISS instance.
│   │
│   └── JWST.json
│       # Official ESA problem definition for the JWST instance.
│
├── results/
│   # Output logs and best solutions obtained from GA executions,
│   # stored separately per problem instance.
│
├── main.py
│   # Entry point of the project:
│   # parses input arguments, selects the problem instance,
│   # initializes the UDP, and runs the Genetic Algorithm.
│
├── .gitignore
│   # Git configuration file:
│   # excludes cache files, virtual environments, IDE settings,
│   # and optionally generated results.
│
├── requirements.txt
│   # External Python dependencies (numpy, numba, matplotlib)
│
└── README.md
    # Project documentation:
    # describes the problem, methodology, structure, and results.
```

## Execution

Requirements:

Python 3.9+  
NumPy   
Numba (optional)

Run example:
```
python main.py --problem ISS
python main.py --problem JWST
python main.py --problem Enterprise
```
### Results

Observed behavior:

ISS: fast convergence and stable solutions         
JWST: gradual improvement          
Enterprise: longest runtime and frequent plateaus

The results confirm that problem complexity directly impacts convergence speed and optimization difficulty.
Valid solutions were obtained for all instances.
