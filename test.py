import numpy as np
import math


y1 = 3*math.sin(5*math.pi/8)
y2 = 3*math.sin(5*math.pi/6)
y3 = 3*math.sin(5*math.pi/5)
y4 = 3*math.sin(5*math.pi/3)
y5 = 3*math.sin(5*math.pi)
print(y1, y2, y3, y4, y5)

x = np.linspace(4*math.pi, 4.3*math.pi, 30)
with open('data_inputs.txt', 'a') as the_file:
    for item in x:
        y = 3 * (math.sin(5 * item))
        the_file.write(f"{item}\t{y}\n")
