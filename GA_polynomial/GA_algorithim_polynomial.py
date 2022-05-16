import pygad
import numpy as np
import matplotlib.pyplot as plt
import math
import time

start = time.time()

"""
Given the following function:
    y = f(w1:w3) = w1x^2 + w2x + w3
What are the best values for the 3 weights (w1 to w3)? 
"""

function_inputs, desired_output = [], []
# read data from the file
with open('data_polynomial.txt') as f:
    for line in f:
        function_inputs.append(float(line.split()[0]))
        desired_output.append(float(line.split()[1]))


def fitness_func(solution, solution_idx):
    output, temp_fitness = 0, 0

    for each in range(len(function_inputs)):
        output = (solution[0] * math.pow(function_inputs[each], 2)) + (solution[1] * function_inputs[each]) + solution[2]
        temp_fitness += 1.0 / np.abs(output - desired_output[each])
    fitness = temp_fitness / 6
    return fitness


fitness_function = fitness_func
num_generations = 150
num_parents_mating = 7
parent_selection_type = "tournament"
sol_per_pop = 75
num_genes = 3
last_fitness = 0


def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       on_generation=callback_generation,
                       parent_selection_type=parent_selection_type)


ga_instance.run()

end = time.time()
ga_instance.plot_fitness(xlabel="Generacje", title="")
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

desired_output_prc, output_for_prc, prc = 0, 0, 0
for each in range(len(function_inputs)):
    output_for_prc = np.abs((solution[0] * math.pow(function_inputs[each], 2)) + (solution[1] * function_inputs[each]) + solution[2])
    desired_output_prc = np.abs((4 * math.pow(function_inputs[each], 2)) + (-5 * function_inputs[each]) + 3)
    if output_for_prc > desired_output_prc:
        prc = 100 - ((desired_output_prc / output_for_prc) * 100)
    else:
        prc = 100 - ((output_for_prc / desired_output_prc) * 100)
print("Error rate of fitness: {:.2f}%".format(prc))

x = np.linspace(-25, 25, 100)
fx_1, fx_2 = [], []
for i in x:
    fx_1.append((4 * math.pow(i, 2)) + (-5 * i) + 3)
    fx_2.append((solution[0] * math.pow(i, 2)) + (solution[1] * i) + solution[2])

fig, ax1 = plt.subplots(sharex=True)
fig.suptitle('Wykres funkcji zadanej do obliczonej')
ax1.plot(x, fx_1, x, fx_2)
plt.show()

print("Program runtime: {:.2f} s".format(end - start))

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Wykres funkcji zadanej do obliczonej')
ax1.plot(x, fx_1)
ax2.plot(x, fx_2)
plt.show()

end = time.time()
print(end - start)

