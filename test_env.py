import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import hashlib
import pandas as pd
import itertools


def read_data(path):
    df = pd.read_csv(path, encoding='unicode_escape')
    df[['Sila', 'Przemieszczenie']] = df['Sila;Przemieszczenie'].str.split(';', expand=True)
    df['Sila'] = pd.to_numeric(df['Sila'])
    df['Przemieszczenie'] = pd.to_numeric(df['Przemieszczenie'])
    desired_output_FORCE = df['Sila'].to_list()
    function_inputs_DISPLACEMENTS = df['Przemieszczenie'].to_list()
    return desired_output_FORCE, function_inputs_DISPLACEMENTS


def main(desired_output_FORCE=[], function_inputs_DISPLACEMENTS=[] ):
    window = make_window(sg.theme())

    # This is an Event Loop
    while True:
        event, values = window.read(timeout=100)
        # keep an animation running so show things are happening
        if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
            print('============ Event = ', event, ' ==============')
        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break
        elif event == 'About':
            print("[LOG] Clicked About!")
            sg.popup('Frontend PySimpleGUI',
                     'Backend PyGAD',
                     'Right click anywhere to see right click menu',
                     'Visit each of the tabs to see available elements',
                     'Output of event and values can be see in Output tab',
                     'The event and values dictionary is printed after every event', keep_on_top=True)
        elif event == 'Popup':
            print("[LOG] Clicked Popup Button!")
            sg.popup("You pressed a button!", keep_on_top=True)
            print("[LOG] Dismissing Popup!")
        elif event == 'Plot_1':
            # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
            plt.figure(1)
            fig = plt.gcf()
            DPI = fig.get_dpi()
            # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
            fig.set_size_inches(604 * 2 / float(DPI), 604 / float(DPI))
            # -------------------------------
            x = np.linspace(0, 2 * np.pi)
            y = np.sin(x)
            plt.plot(x, y)
            plt.title('y=sin(x)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid()
            draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            print("[LOG] Draw a plot_1!")
        elif event == "Open File":
            print("[LOG] Clicked Open File!")
            folder_or_file = sg.popup_get_file('Choose your file', keep_on_top=True)
            sg.popup("You chose: " + str(folder_or_file), keep_on_top=True)
            read_data(folder_or_file)
            print(desired_output_FORCE)
            window['-INPUT_SHOW-'].print('desired output FORCE: ', 2*'\n', desired_output_FORCE, '\n', 'Data type: ' + str(type(desired_output_FORCE[0])), 2*'\n', 'function inputs DISPLACEMENTS: ', 2*'\n', function_inputs_DISPLACEMENTS, '\n', 'Data type: ' + str(type(function_inputs_DISPLACEMENTS[0])))
            print("[LOG] User chose file: " + str(folder_or_file))
        elif event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Versions':
            sg.popup(sg.get_versions(), keep_on_top=True)
        elif event == 'SUBMIT':
            print('-------- Values Dictionary (key=value) --------')
            for key in values:
                print(key, ' = ', values[key])
                sol_per_pop = values[0]
    window.close()
    exit(0)