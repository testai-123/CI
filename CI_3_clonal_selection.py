'''
Implementation of Clonal selection algorithm using Python.
'''

!pip install numpy

import numpy as np

def initialize_population(size, dimension):
    return np.random.rand(size, dimension)

def evaluate_fitness(population, target_function):
    return np.array([target_function(ind) for ind in population])

def select_top_clones(population, fitness, num_clones):
    sorted_indices = np.argsort(fitness)
    return population[sorted_indices[:num_clones]]

def mutate_clones(clones, mutation_rate):
    mutations = np.random.normal(0, mutation_rate, clones.shape)
    return clones + mutations

def clonal_selection_algorithm(target_function, population_size=50, clone_size=10, mutation_rate=0.1, generations=100):
    dimension = 2  # Example for 2D optimization
    population = initialize_population(population_size, dimension)
    
    for _ in range(generations):
        fitness = evaluate_fitness(population, target_function)
        clones = select_top_clones(population, fitness, clone_size)
        mutated_clones = mutate_clones(clones, mutation_rate)
        
        population[:clone_size] = mutated_clones  # Replace worst with best clones
    
    best_index = np.argmin(evaluate_fitness(population, target_function))
    return population[best_index]

# Example target function (minimizing distance to [0, 0])
def target_function(ind):
    return np.sum(ind**2)

best_solution = clonal_selection_algorithm(target_function)
print("Best solution found:", best_solution)
