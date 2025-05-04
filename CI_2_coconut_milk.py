'''
Optimization of genetic algorithm parameter in hybrid genetic algorithm-neural network
modelling: Application to spray drying of coconut milk.
'''

!pip install numpy
!pip install scikit-learn
!pip install deap


import numpy as np
from sklearn.neural_network import MLPRegressor
from deap import base, creator, tools, algorithms


def create_nn_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(individual):
    inlet_temp, feed_flow, atomization_pressure = individual
    predicted_output = nn_model.predict([[inlet_temp, feed_flow, atomization_pressure]])
    fitness = predicted_output[0]
    return fitness,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 100, 200)  # Inlet temperature range
toolbox.register("attr_flow", np.random.uniform, 10, 50)     # Feed flow rate range
toolbox.register("attr_pressure", np.random.uniform, 1, 5)   # Atomization pressure range
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float, toolbox.attr_flow, toolbox.attr_pressure), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


X_train = np.array([[150, 30, 3], [160, 35, 4], [170, 40, 2]]) 
y_train = np.array([0.85, 0.90, 0.80]) 
nn_model = create_nn_model(X_train, y_train)


population = toolbox.population(n=50)
ngen, cxpb, mutpb = 40, 0.7, 0.2
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
print("Best Parameters:", best_individual)
print("Predicted Powder Yield:", evaluate(best_individual)[0])