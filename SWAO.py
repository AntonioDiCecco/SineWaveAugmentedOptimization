import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function
def rosenbrock(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

# Differential Evolution from Scratch
def differential_evolution(iterations, population_size=15, F=0.8, CR=0.9):
    population = np.random.uniform(-5, 5, (population_size, 2))
    for _ in range(iterations):
        new_population = np.copy(population)
        for i in range(population_size):
            targets = [j for j in range(population_size) if j != i]
            a, b, c = population[np.random.choice(targets, 3, replace=False)]

            # Mutation
            mutant = a + F * (b - c)

            # Crossover
            trial = np.where(np.random.rand(2) < CR, mutant, population[i])

            # Selection
            if rosenbrock(trial) < rosenbrock(population[i]):
                new_population[i] = trial

        population = new_population

    best_individual = min(population, key=rosenbrock)
    return rosenbrock(best_individual)


# Particle Swarm Optimization
def particle_swarm(iterations, population_size=15):
    population = np.random.uniform(-5, 5, (population_size, 2))
    velocity = np.zeros_like(population)
    local_best = np.copy(population)
    global_best = min(population, key=rosenbrock)
    for _ in range(iterations):
        # Update velocity and position
        r1, r2 = np.random.rand(), np.random.rand()
        velocity = 0.5 * velocity + 1.41 * r1 * (local_best - population) + 1.41 * r2 * (global_best - population)
        population += velocity
        # Update local and global bests
        for i in range(population_size):
            if rosenbrock(population[i]) < rosenbrock(local_best[i]):
                local_best[i] = population[i]
        global_best = min(local_best, key=rosenbrock)
    return rosenbrock(global_best)

# SineWaveAugmentedOptimization (SWAO) with detailed comments
def SineWaveAugmentedOptimization(iterations, population_size=15):
    # Initialize population
    population = np.random.uniform(-5, 5, (population_size, 2))

    # Iterate through each generation
    for _ in range(iterations*population_size):

        fitness_values = np.array([rosenbrock(point) for point in population])         # Calculate fitness values for the population
        worst_index = np.argmax(fitness_values)          # Identify the worst and best individuals based on their fitness
        best_index = np.argmin(fitness_values)
        sorted_indices = np.argsort(fitness_values)         # Sort population indices based on fitness values
        top_half_indices = sorted_indices[:population_size//2]         # Select the top half of the population based on fitness
        random_index = np.random.choice(top_half_indices)         # Select a random individual from the top half of the population
        diff = np.abs(population[best_index] - population[worst_index])         # Compute the difference vector between the best and worst individuals
        theta = np.random.uniform(0, 2 * np.pi, 2)         # Generate a random angle theta for sine wave modulation
        new_point = 0.5 * population[best_index] + 0.5 * population[random_index] + np.sin(theta) * diff          # Create a new individual by mixing and modulating the existing individuals

        new_fitness = rosenbrock(new_point)

        if new_fitness < fitness_values[worst_index]:  # Replace the worst individual if the new individual is better
            population[worst_index] = new_point

    return fitness_values.min()

# Comparison
iterations = 1000
population_size = 15
runs = 100

fail_de, fail_ga, fail_pso, fail_sine = 0, 0, 0, 0

for _ in range(runs):
    if differential_evolution(iterations) >= 0.1:
        fail_de += 1
    if particle_swarm(iterations) >= 0.1:
        fail_pso += 1
    if SineWaveAugmentedOptimization(iterations) >= 0.1:
        fail_sine += 1

print(f"Differential Evolution failed {fail_de} times out of {runs} runs.")
print(f"Particle Swarm failed {fail_pso} times out of {runs} runs.")
print(f"SineWaveAugmentedOptimization failed {fail_sine} times out of {runs} runs.")
