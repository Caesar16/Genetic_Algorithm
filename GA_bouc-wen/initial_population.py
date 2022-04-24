mport pygad
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
import mpmath as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")



start = time.time()
"""
Given the following function:
    y = f(w1:w6) = w1*sin(w2*x)
What are the best values for the 2 weights (w1, w2)? 
"""
# read data from the file
df = pd.read_csv('./bialy_c109-1.csv', encoding= 'unicode_escape')
df[['Sila', 'Przemieszczenie']] = df['Sila;Przemieszczenie'].str.split(';', expand=True)
df['Sila'] = pd.to_numeric(df['Sila'])
df['Przemieszczenie'] = pd.to_numeric(df['Przemieszczenie'])
desired_output_FORCE = df['Sila'].to_list()
function_inputs_DISPLACEMENTS = df['Przemieszczenie'].to_list()

for n, each in enumerate(range(len(desired_output_FORCE))):
    print(n, type(function_inputs_DISPLACEMENTS[each]), desired_output_FORCE[each])

fig, ax = plt.subplots(sharex=True)
fig.suptitle('Wykres funkcji zadanej do obliczonej')
plt.plot(function_inputs_DISPLACEMENTS, desired_output_FORCE)
plt.show()