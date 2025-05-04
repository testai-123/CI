'''
Implement DEAP (Distributed Evolutionary Algorithms) using Python.
'''


!pip install deap


from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt

# Define evaluation function
def eval_func(individual):
    x = individual[0]
    return x * np.sin(10 * np.pi * x) + 1.0,

# Problem setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=50)
NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

top_individual = tools.selBest(population, k=1)[0]
print("Best Solution: ", top_individual, "Fitness: ", top_individual.fitness.values[0])
