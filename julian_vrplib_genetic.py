import random, math, numpy as np
import matplotlib.pyplot as plt
from numba import jit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

@jit(nopython=True)
def route_cost_numba(routes_flat, route_lengths, matrix):
    """Numba-optimized route cost calculation"""
    total = 0.0
    idx = 0
    for length in route_lengths:
        for i in range(length - 1):
            total += matrix[routes_flat[idx + i]][routes_flat[idx + i + 1]]
        idx += length
    return total

def route_cost(routes, matrix):
    """Convert routes to flat format and calculate cost"""
    routes_flat = np.array([node for route in routes for node in route], dtype=np.int32)
    route_lengths = np.array([len(route) for route in routes], dtype=np.int32)
    return route_cost_numba(routes_flat, route_lengths, matrix)

@jit(nopython=True)
def two_opt_numba(route, matrix):
    """Numba-optimized 2-opt"""
    n = len(route)
    if n <= 4:
        return route
    
    improved = True
    iterations = 0
    max_iterations = 100
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b, c, d = route[i-1], route[i], route[j], route[j+1]
                current_cost = matrix[a, b] + matrix[c, d]
                new_cost = matrix[a, c] + matrix[b, d]
                if new_cost < current_cost - 0.001:
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    return route

# ----------------------------
#  Distance + cost functions
# ----------------------------
def generate_adjacency_matrix(node_coords):
    coords = np.array(node_coords, dtype=np.float32)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    matrix = np.sqrt(np.sum(diff**2, axis=2)).astype(np.float32)
    return matrix

# ----------------------------
#  Fast Initialization
# ----------------------------
def nearest_neighbor_solution_fast(customers, demands, capacity, matrix, depot=0):
    """Fast nearest neighbor using numpy"""
    unvisited = set(customers)
    routes = []
    
    while unvisited:
        route = [depot]
        load = 0
        current = depot
        
        while unvisited:
            # Vectorized distance calculation
            candidates = [c for c in unvisited if load + demands[c] <= capacity]
            if not candidates:
                break
            
            # Find nearest
            dists = matrix[current, candidates]
            best_idx = np.argmin(dists)
            best_customer = candidates[best_idx]
            
            route.append(best_customer)
            load += demands[best_customer]
            current = best_customer
            unvisited.remove(best_customer)
        
        route.append(depot)
        routes.append(route)
    
    return routes

def init_population(pop_size, customers, num_vehicles, demands, capacity, depot, matrix):
    population = []
    
    # Add 3 nearest neighbor solutions (reduced from 5)
    for _ in range(min(3, pop_size // 5)):
        shuffled = customers[:]
        random.shuffle(shuffled)
        nn_routes = nearest_neighbor_solution_fast(shuffled, demands, capacity, matrix, depot)
        population.append(nn_routes)
    
    # Fill rest with random solutions
    while len(population) < pop_size:
        shuffled = customers[:]
        random.shuffle(shuffled)
        population.append(rebuild_routes(shuffled, num_vehicles, demands, capacity, depot))
    
    return population

# ----------------------------
#  Selection (tournament) - inlined
# ----------------------------
@jit(nopython=True)
def tournament_select(costs, k=4):
    """Select k random indices and return the best one"""
    n = len(costs)
    best_idx = np.random.randint(0, n)
    best_cost = costs[best_idx]
    
    for _ in range(k - 1):
        idx = np.random.randint(0, n)
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_idx = idx
    
    return best_idx

# ----------------------------
#  Order Crossover (OX) - optimized
# ----------------------------
def crossover(parent1, parent2):
    # Flatten routes (exclude depot 0)
    p1 = [c for route in parent1 for c in route if c != 0]
    p2 = [c for route in parent2 for c in route if c != 0]
    n = len(p1)
    if n < 2:
        return p1[:]

    # Select two cut points
    a, b = sorted(random.sample(range(n), 2))

    # Check if subsequences are identical
    if p1[a:b] == p2[a:b]:
        return p1[:]

    # Execute OX crossover
    child = [None] * n
    child[a:b] = p1[a:b]
    child_set = set(p1[a:b])
    ptr = 0
    for c in p2:
        if c not in child_set:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = c
    return child

# ----------------------------
#  Mutation - simplified and faster
# ----------------------------
def mutate(customers, mutation_rate=0.3, gen=0, generations=2000):
    n = len(customers)
    # Adaptive mutation rate
    progress = gen / generations
    adaptive_rate = mutation_rate * (1.0 - 0.5 * progress)
    
    # Swap mutation
    if random.random() < adaptive_rate:
        i, j = random.sample(range(n), 2)
        customers[i], customers[j] = customers[j], customers[i]
    
    # 2-opt style inversion
    if random.random() < adaptive_rate * 0.5:
        i, j = sorted(random.sample(range(n), 2))
        customers[i:j] = reversed(customers[i:j])
    
    return customers

# ----------------------------
#  Route rebuild - optimized
# ----------------------------
def rebuild_routes(customers, num_vehicles, demands, capacity, depot=0):
    routes = []
    current = [depot]
    load = 0
    
    for c in customers:
        d = demands[c]
        if load + d > capacity:
            current.append(depot)
            routes.append(current)
            current = [depot]
            load = 0
        current.append(c)
        load += d
    
    if len(current) > 1:
        current.append(depot)
        routes.append(current)
    
    # Add empty routes if needed
    while len(routes) < num_vehicles:
        routes.append([depot, depot])
    
    return routes

# ----------------------------
#  Local improvement - using numba
# ----------------------------
def local_improvement_fast(routes, matrix):
    """Fast local improvement using numba"""
    improved_routes = []
    for route in routes:
        if len(route) > 4:
            route_array = np.array(route, dtype=np.int32)
            improved = two_opt_numba(route_array, matrix)
            improved_routes.append(improved.tolist())
        else:
            improved_routes.append(route[:])
    return improved_routes

# ----------------------------
#  Inter-route optimization - simplified
# ----------------------------
def relocate_customer_fast(routes, matrix, demands, capacity, max_tries=50):
    """Faster relocate with limited tries"""
    improved = False
    tries = 0
    
    route_indices = list(range(len(routes)))
    random.shuffle(route_indices)
    
    for i in route_indices[:min(5, len(routes))]:
        if len(routes[i]) <= 3:
            continue
            
        for j in route_indices[:min(5, len(routes))]:
            if i == j or tries >= max_tries:
                continue
            
            tries += 1
            
            for pos_i in range(1, len(routes[i]) - 1):
                customer = routes[i][pos_i]
                demand = demands[customer]
                
                # Check capacity
                load_j = sum(demands[routes[j][k]] for k in range(1, len(routes[j]) - 1))
                if load_j + demand > capacity:
                    continue
                
                # Try best insertion position only
                best_pos = 1
                best_delta = float('inf')
                
                for pos_j in range(1, len(routes[j])):
                    old_cost = (matrix[routes[i][pos_i-1], customer] + 
                               matrix[customer, routes[i][pos_i+1]])
                    new_cost_i = matrix[routes[i][pos_i-1], routes[i][pos_i+1]]
                    
                    old_cost += matrix[routes[j][pos_j-1], routes[j][pos_j]]
                    new_cost_j = (matrix[routes[j][pos_j-1], customer] + 
                                 matrix[customer, routes[j][pos_j]])
                    
                    delta = (new_cost_i + new_cost_j) - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos_j
                
                if best_delta < -0.01:
                    routes[i].pop(pos_i)
                    routes[j].insert(best_pos, customer)
                    return routes, True
    
    return routes, improved

# ----------------------------
#  Batch cost calculation
# ----------------------------
def batch_route_cost(population, matrix):
    """Calculate costs for entire population efficiently"""
    return np.array([route_cost(sol, matrix) for sol in population], dtype=np.float32)

# ----------------------------
#  Main GA loop (highly optimized)
# ----------------------------
def genetic_algorithm(instance, pop_size=200, generations=2000,
                      mutation_rate=0.35, elitism=10, tournament_k=4, 
                      target_gap=0.07, optimal_cost=None):
    coords_raw = instance["node_coord"]
    node_coords = list(coords_raw.values()) if isinstance(coords_raw, dict) else coords_raw
    demands_raw = instance["demand"]
    demands = list(demands_raw.values()) if isinstance(demands_raw, dict) else demands_raw
    capacity = instance["capacity"]
    depot = 0
    customers = list(range(1, len(node_coords)))
    total_demand = sum(demands[c] for c in customers)
    num_vehicles = max(instance.get("vehicles", 5), math.ceil(total_demand / capacity))
    matrix = generate_adjacency_matrix(node_coords)
    
    # Get optimal cost if provided
    if optimal_cost is None:
        optimal_cost = instance.get("optimal_cost", None)

    # Initialize population
    population = init_population(pop_size, customers, num_vehicles, demands, capacity, depot, matrix)
    
    # Calculate initial costs
    costs = batch_route_cost(population, matrix)
    best_idx = np.argmin(costs)
    best = population[best_idx]
    best_cost = costs[best_idx]
    
    print(f"Initial cost: {best_cost:.2f}")
    
    # Calculate initial gap if optimal cost is known
    if optimal_cost is not None:
        initial_gap = (best_cost - optimal_cost) / optimal_cost * 100
        print(f"Initial gap: {initial_gap:.2f}%")
    
    stagnation_counter = 0

    for gen in range(generations):
        new_pop = []
        
        # Sort population by cost
        sorted_indices = np.argsort(costs)
        
        # Extract elites
        elite_indices = sorted_indices[:elitism]
        elites = [population[i] for i in elite_indices]
        
        # Apply local search to elites periodically
        if gen % 30 == 0:
            elites = [local_improvement_fast([route[:] for route in sol], matrix) for sol in elites]
        
        new_pop.extend(elites)

        # Generate offspring
        while len(new_pop) < pop_size:
            # Fast tournament selection
            p1_idx = tournament_select(costs, tournament_k)
            p2_idx = tournament_select(costs, tournament_k)
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            child_seq = crossover(p1, p2)
            child_seq = mutate(child_seq, mutation_rate, gen, generations)
            child_routes = rebuild_routes(child_seq, num_vehicles, demands, capacity, depot)
            
            # Apply local improvement less frequently
            if gen < 300 or gen % 5 == 0:
                child_routes = local_improvement_fast(child_routes, matrix)
            
            # Apply inter-route optimization rarely
            if gen % 20 == 0 and random.random() < 0.2:
                child_routes, _ = relocate_customer_fast(child_routes, matrix, demands, capacity)
            
            new_pop.append(child_routes)

        population = new_pop
        costs = batch_route_cost(population, matrix)
        
        current_best_idx = np.argmin(costs)
        current_best_cost = costs[current_best_idx]
        
        if current_best_cost < best_cost - 0.01:
            best = population[current_best_idx]
            best_cost = current_best_cost
            stagnation_counter = 0
            
            # Check if we've reached target gap
            if optimal_cost is not None:
                current_gap = (best_cost - optimal_cost) / optimal_cost
                if current_gap <= target_gap:
                    gap_percentage = current_gap * 100
                    print(f"\n🎯 Target gap of {target_gap*100:.1f}% achieved!")
                    print(f"Gen {gen}: best cost {best_cost:.2f}, gap: {gap_percentage:.2f}%")
                    break
        else:
            stagnation_counter += 1
        
        # Print progress
        if gen % 100 == 0:
            if optimal_cost is not None:
                current_gap = (best_cost - optimal_cost) / optimal_cost * 100
                print(f"Gen {gen}: best cost {best_cost:.2f}, gap: {current_gap:.2f}%")
            else:
                print(f"Gen {gen}: best cost {best_cost:.2f}")
        
        # Diversity injection if stagnated
        if stagnation_counter > 150:
            print(f"Gen {gen}: Injecting diversity...")
            num_random = pop_size // 5
            for idx in range(num_random):
                shuffled = customers[:]
                random.shuffle(shuffled)
                population[-(idx+1)] = rebuild_routes(shuffled, num_vehicles, demands, capacity, depot)
            costs = batch_route_cost(population, matrix)
            stagnation_counter = 0

    # Final intensive local search
    print("Applying final optimization...")
    for _ in range(3):
        best = local_improvement_fast(best, matrix)
        improved = True
        tries = 0
        while improved and tries < 3:
            best, improved = relocate_customer_fast(best, matrix, demands, capacity)
            tries += 1
    
    final_cost = route_cost(best, matrix)
    
    # Calculate final gap
    if optimal_cost is not None:
        final_gap = (final_cost - optimal_cost) / optimal_cost * 100
        print(f"\nFinal cost: {final_cost:.2f}")
        print(f"Optimal cost: {optimal_cost:.2f}")
        print(f"Final gap: {final_gap:.2f}%")
    else:
        print(f"\nFinal cost: {final_cost:.2f}")
    
    coords_dict = dict(enumerate(node_coords))
    #visualize_routes(best, coords_dict, title="Best Routes Found by GA")
    return {"route": best, "cost": final_cost}

#def visualize_path(best_route, coords, title="Optimal path"):
    x = [coords[i][0] for i in best_route]
    y = [coords[i][1] for i in best_route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', linewidth=2, markersize=8)
    for i, (xi, yi) in coords.items():
        plt.text(xi + 0.3, yi + 0.3, str(i), fontsize=9)

    plt.scatter(x[0], y[0], color='green', s=100, label="Start")
    plt.scatter(x[-1], y[-1], color='red', s=100, label="End")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

#def visualize_routes(routes, coords, title="Road VRP"):
   # plt.figure(figsize=(10, 8))
    # for idx, route in enumerate(routes):
       # if len(route) > 2:
           # x = [coords[i][0] for i in route]
           # y = [coords[i][1] for i in route]
           # plt.plot(x, y, 'o-', linewidth=2, markersize=8, 
                   # color=colors[idx], label=f"Road {idx+1}")
    
    #for i, (xi, yi) in coords.items():
        #plt.text(xi + 0.5, yi + 0.5, str(i), fontsize=9)
    
    #depot_x, depot_y = coords[0]
    #plt.scatter(depot_x, depot_y, color='red', s=200, marker='s', 
               #label="Depot", zorder=5, edgecolors='black', linewidths=2)
    
    #plt.title(title)
    #plt.legend()
    #plt.grid(True)
    #plt.axis('equal')
    #plt.show()
