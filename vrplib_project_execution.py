import sys, os , time
sys.path.append(os.path.dirname(__file__))

import vrplib
from julian_vrplib_genetic import genetic_algorithm

# Load instance and optimal solution
instance = vrplib.read_instance("C101.vrp")
optimal_sol = vrplib.read_solution("C101.sol")

# time_window = instance['time_window']

# Extract earliest and latest lists
# earliest = list(time_window[:, 0])
# latest = list(time_window[:, 1])

# Run your algorithm
start = time.time()
# my_solution = genetic_algorithm(instance,earliest=earliest,latest=latest)
my_solution = genetic_algorithm(instance)
end = time.time() - start
best_route = my_solution["route"]
best_cost = my_solution["cost"]

gap = 100 * (my_solution["cost"] - optimal_sol["cost"]) / optimal_sol["cost"]
print(f"Execution time: {end:.3f} seconds")
print(f"Best cost found: {best_cost:.2f}")
print(f"Number of routes: {len(my_solution["route"])}")
print(f"Reference (optimal) cost: {optimal_sol['cost']:.2f}")
print(f"Gap vs reference: {gap:.2f}%")
