import matplotlib.pyplot as plt


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
