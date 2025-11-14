# End-to-End CVRP Optimization Using Genetic Algorithms

This project implements a complete optimization pipeline for solving the **Capacitated Vehicle Routing Problem (CVRP)** using a **Genetic Algorithm**.  
It includes modeling, implementation, visualization, testing, and analysis.

---

##  Problem Description
The Capacitated Vehicle Routing Problem aims to determine the optimal set of delivery routes for a fleet of vehicles with limited capacity, minimizing the total distance traveled while serving all customers.

---

##  Project Workflow

### 1. **Modeling the CVRP**
- Defined customer nodes, demands, and depot.
- Specified vehicle capacities.
- Implemented objective function and constraint structure.

### 2. **Genetic Algorithm Implementation**
- Chromosome representation of customer sequences.
- Fitness evaluation (route distance + feasibility checks).
- Roulette/tournament selection.
- Ordered/PMX crossover.
- Mutation and repair operators.
- Stopping criteria and best-solution tracking.

### 3. **Visualization**
- Route plots using matplotlib.
- Convergence curve visualization.
- Comparative plots between GA configurations.

### 4. **Testing & Analysis**
- Evaluated distances across generations.
- Tested multiple mutation and crossover rates.
- Analyzed runtime, solution stability, and improvements.

---

##  Results
- Successfully generated efficient delivery routes respecting capacity constraints.
- Improved total distance across generations.
- Visualization of final best route included in `/plots/`.

---

##  Tech Stack
- Python  
- NumPy  
- Matplotlib  
- Heuristics & Genetic Algorithms  
- Operations Research / Optimization

---

##  Structure
