import math
import matplotlib.pyplot as plt
import numpy as np

function_inputs, desired_output = [], []
# read data from the file
with open('data_sinus.txt') as f:
    for line in f:
        function_inputs.append(float(line.split()[0]))
        desired_output.append(float(line.split()[1]))


fitness_1, fitness_2 = 0, 0
for each in range(len(function_inputs)):
    output = 3.78 * (math.sin(4.9 * function_inputs[each]))
    output_2 = 6.26 * (math.sin(1.98 * function_inputs[each]))
    print(output, output_2, desired_output[each])
    fitness_1 += 1.0 / abs((output - desired_output[each]))
    fitness_2 += 1.0 / abs((output_2 - desired_output[each]))
print(fitness_1, fitness_2)


x = np.linspace(-math.pi, math.pi, 50)
fx_1, fx_2, fx_3 = [], [], []
for i in range(len(x)):
    fx_1.append(3.78*(math.sin(4.9 * x[i])))
    fx_2.append(3 * (math.sin(5 * x[i])))
    fx_3.append(6.26 * (math.sin(1.98 * x[i])))

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, fx_3, x, fx_2)
ax2.plot(x, fx_1, x, fx_2)
plt.show()

















import pygad
import numpy as np
import matplotlib.pyplot as plt
import math
import time

"""
Given the following function:
    y = f(w1:w6) = w1*sin(w2*x)
What are the best values for the 2 weights (w1, w2)? 
"""

start = time.time()

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
        temp_fitness += 1.0 / abs((output - desired_output[each]))
    fitness = temp_fitness
    return fitness




fitness_function = fitness_func

num_generations = 1000 # Number of generations.
num_parents_mating = 10 #Number of solutions to be selected as parents in the mating pool.
parent_selection_type = "tournament"

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 150  # Number of solutions in the population.
num_genes = 2


last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
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

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

#prediction = np.sum(np.array(function_inputs)*solution)
#print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
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