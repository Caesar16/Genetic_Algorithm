import pygad
import numpy as np
import matplotlib.pyplot as plt
import math
import time

start = time.time()

"""
Given the following function:
    y = f(w1:w6) = w1*sin(w2*x)
What are the best values for the 2 weights (w1, w2)? 
"""

function_inputs, desired_output = [], []
# read data from the file
with open('data_sinus.txt') as f:
    for line in f:
        function_inputs.append(float(line.split()[0]))
        desired_output.append(float(line.split()[1]))


def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    output, temp_fitness = 0, 0
    for each in range(len(function_inputs)):
        output = solution[0]*(math.sin(solution[1] * function_inputs[each]))
        temp_fitness += abs((output - desired_output[each]))
    fitness = 100 / temp_fitness
    return fitness


fitness_function = fitness_func
num_generations = 500
num_parents_mating = 10
parent_selection_type = "tournament"
sol_per_pop = 150
num_genes = 2
last_fitness = 0


def callback_generation(ga_instance):
    global last_fitness


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       on_generation=callback_generation,
                       parent_selection_type=parent_selection_type,
                       init_range_low=-10,
                       init_range_high=10,
                       )


ga_instance.run()

ga_instance.plot_fitness()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations." \
    .format(best_solution_generation=ga_instance.best_solution_generation))

filename = 'genetic'
ga_instance.save(filename=filename)
loaded_ga_instance = pygad.load(filename=filename)

x = np.linspace(-2*math.pi, 2*math.pi, 300)
fx_1, fx_2 = [], []
for i in range(len(x)):
    fx_1.append(solution[0]*(math.sin(solution[1] * x[i])))
    fx_2.append(3 * (math.sin(5 * x[i])))

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, fx_1, x, fx_2)
plt.show()

end = time.time()
print(end - start)
