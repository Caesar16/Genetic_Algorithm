import math
import matplotlib.pyplot as plt
import numpy as np

function_inputs, desired_output = [], []
# read data from the file
with open('data_sinus.txt') as f:
    for line in f:
        function_inputs.append(float(line.split()[0]))
        desired_output.append(float(line.split()[1]))

diff_1, diff_2 = 0, 0
for each in range(len(function_inputs)):
    output = -3.15 * (math.sin(-4.97 * function_inputs[each]))
    output_2 = -2.71 * (math.sin(-5.59 * function_inputs[each]))
    diff_1 += abs(output - desired_output[each])
    diff_2 += abs(output_2 - desired_output[each])
    print(round(diff_1, 3), round(diff_2, 3))
print(round(1.0 / diff_1, 3), round(1.0 / diff_2, 3))

x = np.linspace(-math.pi, math.pi, 300)
fx_1, fx_2, fx_3 = [], [], []
for i in range(len(x)):
    fx_1.append(-3.15*(math.sin(-4.97 * x[i])))
    fx_2.append(3 * (math.sin(5 * x[i])))
    fx_3.append(-2.71 * (math.sin(-5.59 * x[i])))

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Wykresy funkcji z parametrami z epoki 2 i 7, na tle funkcji szukanej')
ax1.plot(x, fx_3, x, fx_2)
ax2.plot(x, fx_1, x, fx_2)
plt.show()













