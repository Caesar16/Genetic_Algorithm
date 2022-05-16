import matplotlib.pyplot as plt
import pygad
import numpy as np
import math
import time

start = time.time()


def GeneticOptimization(function_inputs, desired_output, sol_per_pop=100, mutation_probability=0.5,
                        num_generations=10, crossover_probability=0.5, num_parents_mating=20, crossover_type="single_point",
                        parent_selection_type="sss", mutation_type="random", keep_parents=-1, mutation_by_replacement=False,
                        K_tournament=3, random_mutation_min_val=-1.0, random_mutation_max_val=1.0, stop_criteria=None):

    print(num_parents_mating)
    global last_fitness
    global_cache, local_cache = [], []

    def draw_chart(function_inputs, desired_output, solution):
        desired_output_prc, output_for_prc, prc = 0, 0, 0
        for each in range(len(function_inputs)):
            output_for_prc = np.abs(solution[0] * (math.sin(solution[1] * function_inputs[each])))
            desired_output_prc = np.abs(3 * (math.sin(5 * function_inputs[each])))
            if output_for_prc > desired_output_prc:
                prc = 100 - ((desired_output_prc / output_for_prc) * 100)
            else:
                prc = 100 - ((output_for_prc / desired_output_prc) * 100)
        print("Error rate of fitness: {:.2f}%".format(prc))

        x = np.linspace(-2 * math.pi, 2 * math.pi, 300)
        fx_1, fx_2 = [], []
        for i in range(len(x)):
            fx_1.append(solution[0] * (math.sin(solution[1] * x[i])))
            fx_2.append(3 * (math.sin(5 * x[i])))

        figure, ax1 = plt.subplots(sharex=True)
        figure.suptitle('Wykres funkcji zadanej do obliczonej')
        ax1.plot(x, fx_1, x, fx_2)
        print("Program runtime: {:.2f} s".format(end - start))

        return figure

    def fitness_func(solution, solution_idx):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function calulates the sum of products between each input and its corresponding weight.
        output, temp_fitness = 0, 0
        for each in range(len(function_inputs)):
            output = solution[0] * (math.sin(solution[1] * function_inputs[each]))
            temp_fitness += abs((output - desired_output[each]))
        fitness = 100 / temp_fitness
        return fitness

    fitness_function = fitness_func
    num_genes = 2
    last_fitness = 0

    def callback_generation(ga_instance):
        global last_fitness
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
        last_fitness = ga_instance.best_solution()[1]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=int(num_parents_mating),
                           fitness_func=fitness_function,
                           sol_per_pop=int(sol_per_pop),
                           num_genes=num_genes,
                           on_generation=callback_generation,
                           parent_selection_type=parent_selection_type,
                           init_range_low=-10,
                           init_range_high=10,
                           )

    ga_instance.run()
    fig = ga_instance.plot_fitness(title="Generacje vs. Fitness", xlabel="Generacje")
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations." \
        .format(best_solution_generation=ga_instance.best_solution_generation))
    end = time.time()
    print("Program runtime: {:.2f} s".format(end - start))
    plot = draw_chart(function_inputs, desired_output, solution)

    return fig, plot
