import pygad
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
import mpmath as mp

start = time.time()
global_cache, local_cache = [], []

df = pd.read_csv('bialy_c109-1.csv', encoding='unicode_escape')
df[['Sila', 'Przemieszczenie']] = df['Sila;Przemieszczenie'].str.split(';', expand=True)
df['Sila'] = pd.to_numeric(df['Sila'])
df['Przemieszczenie'] = pd.to_numeric(df['Przemieszczenie'])
desired_output_FORCE = df['Sila'].to_list()
function_inputs_DISPLACEMENTS = df['Przemieszczenie'].to_list()

def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    # Bouc Wen differential equation calculation
    z = 1.0
    v = 0.0
    delta_t = 0.1
    ast = round(solution[0], 3)
    ac = round(solution[7], 3)
    ns = round(solution[1], 3)
    nc = round(solution[8], 3)
    alfas = round(solution[2], 3)
    alfac = round(solution[9], 3)
    betas = round(solution[3], 3)
    betac = round(solution[10], 3)
    gammas = round(solution[4], 3)
    gammac = round(solution[11], 3)
    etas = round(solution[5], 3)
    etac = round(solution[12], 3)
    omegas = round(solution[6], 3)
    omegac = round(solution[13], 3)
    temp_fitness = 0
    for i in range(len(function_inputs_DISPLACEMENTS)):
        if i != (len(function_inputs_DISPLACEMENTS) - 1):
            v = round(((float(function_inputs_DISPLACEMENTS[i]) - float(function_inputs_DISPLACEMENTS[i+1])) / delta_t), 3)
        if v > 0:
            #print(-gammas, abs(v), abs(z), ns - 1, z, betas, v, abs(z), ns, ast, v)
            z_derivative_plus = round(-gammas * abs(v) * mp.power(abs(z), ns - 1) * z - betas * v * mp.power(abs(z), ns) + ast * v, 3)
            z = round((z_derivative_plus * delta_t) + z, 3)
            #print(z_derivative_plus, z)
        else:
            #print(-gammac, abs(v), abs(z), nc - 1, z, betac, v, abs(z), nc, ast, v)
            z_derivative_minus = round((-gammac * abs(v) * mp.power(abs(z), nc - 1) * z) - (betac * v * mp.power(abs(z), nc)) + (ac * v), 3)
            z = round((z_derivative_minus * delta_t) + z, 3)
            #print(z_derivative_minus, z)
        if math.isnan(z):
            z = 0.0000001
        elif math.isinf(z):
            z = 1000000
        if v > 0:
            force = 2 * etas * omegas * v + alfas * mp.power(omegas, 2) * float(function_inputs_DISPLACEMENTS[i]) + (1 - alfas) * mp.power(omegas, 2) * z
        else:
            force = 2 * etac * omegac * v + alfac * mp.power(omegac, 2) * float(function_inputs_DISPLACEMENTS[i]) + (1 - alfac) * mp.power(omegac, 2) * z
        local_cache.append(force)
        temp_fitness += abs(force - float(desired_output_FORCE[i]))
        #print(force)
    new_list = local_cache.copy()
    global_cache.append(new_list)
    local_cache.clear()
    fitness = 10000 / temp_fitness
    return fitness


fitness_function = fitness_func
num_generations = 50
sol_per_pop = 100
num_parents_mating = 10
parent_selection_type = "tournament"
K_tournament = 3
num_genes = 14
last_fitness = 0
keep_parents = -1
stop_criteria = None
mutation_probability = 0.5
mutation_type = "random"
mutation_by_replacement = False
random_mutation_min_val = -1.0
random_mutation_max_val = 1.0
crossover_probability = 0.5
crossover_type = "single_point"
gene_type = float
gene_space = [{"low": 0.001, "high": 0.025, "step": 0.001},
              {"low": 0.43, "high": 10.75, "step": 0.43},
              {"low": 0.07, "high": 1.75, "step": 0.07},
              {"low": 0.1, "high": 2.5, "step": 0.1},
              {"low": 0.3, "high": 7.5, "step": 0.3},
              {"low": 20, "high": 500, "step": 20},
              {"low": 0.3, "high": 7.5, "step": 0.3},
              {"low": 0.05, "high": 1.25, "step": 0.05},
              {"low": 0.2, "high": 5, "step": 0.2},
              {"low": 0.0055, "high": 0.1375, "step": 0.0055},
              {"low": 0.001, "high": 0.025, "step": 0.001},
              {"low": 0.2, "high": 5, "step": 0.2},
              {"low": -575, "high": -23, "step": 20},
              {"low": 1.8, "high": 45, "step": 1.8}]


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
                       parent_selection_type=parent_selection_type,
                       mutation_probability=mutation_probability,
                       crossover_probability=crossover_probability,
                       gene_type=gene_type,
                       gene_space=gene_space,
                       K_tournament=K_tournament,
                       keep_parents=keep_parents,
                       stop_criteria=stop_criteria,
                       mutation_type=mutation_type,
                       mutation_by_replacement=mutation_by_replacement,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val,
                       crossover_type=crossover_type,
                       )

ga_instance.run()
ga_instance.plot_fitness(xlabel="Generacje")
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

print(len(global_cache[solution_idx]))
end = time.time()


def draw_chart(function_inputs, desired_output, count_output):
    prc = 0
    for each in range(len(function_inputs)):
        if float(count_output[each]) > float(desired_output[each]):
            prc = 100 - ((float(desired_output[each]) / float(count_output[each])) * 100)
        else:
            prc = 100 - ((float(count_output[each]) / float(desired_output[each])) * 100)
    print("Error rate of fitness: {:.2f}%".format(prc))

    fig, ax1 = plt.subplots(sharex=True)
    fig.suptitle('Wykres funkcji zadanej do obliczonej')
    ax1.plot(function_inputs, count_output, function_inputs, desired_output)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Wykres funkcji w indywidualnych skalach')
    ax1.plot(function_inputs, count_output)
    ax2.plot(function_inputs, desired_output)
    plt.show()


draw_chart(function_inputs_DISPLACEMENTS, desired_output_FORCE, global_cache[solution_idx])
print("Program runtime: {:.2f} s".format(end - start))