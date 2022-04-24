import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import hashlib
import pandas as pd
import itertools
import GA_backend

'''
     Create a secure login for your scripts without having to include your password in the program. Create an SHA1 hash 
     code for your password using the GUI. Paste into variable in final program
     1. Choose a password
     2. Generate a hash code for your chosen password by running program and entering 'gui' as the password
     3. Type password into the GUI
     4. Copy and paste hash code Window GUI into variable named login_password_hash
     5. Run program again and test your login!
'''


# Use this GUI to get your password's hash code
def HashGeneratorGUI():
    layout = [[sg.T('Password Hash Generator', size=(30, 1), font='Any 15')],
              [sg.T('Password'), sg.In(key='password')],
              [sg.T('SHA Hash'), sg.In('', size=(40, 1), key='hash')],
              ]

    window = sg.Window('SHA Generator', layout, auto_size_text=False, default_element_size=(10, 1),
                       text_justification='r', return_keyboard_events=True, grab_anywhere=False)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            exit(69)

        password = values['password']
        try:
            password_utf = password.encode('utf-8')
            sha1hash = hashlib.sha1()
            sha1hash.update(password_utf)
            password_hash = sha1hash.hexdigest()
            window['hash'].update(password_hash)
        except:
            pass


# ----------------------------- Paste this code into your program / script -----------------------------
# determine if a password matches the secret password by comparing SHA1 hash codes
def PasswordMatches(password, hash):
    password_utf = password.encode('utf-8')
    sha1hash = hashlib.sha1()
    sha1hash.update(password_utf)
    password_hash = sha1hash.hexdigest()
    if password_hash == hash:
        return True
    else:
        return False


def wrong_passwd():
    layout = [[sg.T('Wrong Password', size=(30, 1), font='Any 20')]]
    window = sg.Window('Status', layout, auto_size_text=True, default_element_size=(10, 1),
                       text_justification='c', return_keyboard_events=True, grab_anywhere=False)
    window.read()
    run()


def run():
    login_password_hash = '5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8'
    password = sg.popup_get_text('Enter password', password_char='*')
    if password == 'gui':  # Remove when pasting into your program
        HashGeneratorGUI()  # Remove when pasting into your program
        exit(69)  # Remove when pasting into your program
    if PasswordMatches(password, login_password_hash):
        print('Login SUCCESSFUL')
        custom_meter()
        main()
    else:
        print('Login FAILED!!')
        wrong_passwd()


def custom_meter():
    # layout the form
    layout = [[sg.Text('Initializing')],
              [sg.ProgressBar(300, orientation='h',
                              size=(20, 20), key='progress')],
              [sg.Cancel()]]

    # create the form`
    window = sg.Window('Custom Progress Meter', layout)
    progress_bar = window['progress']
    # loop that would normally do something useful
    for i in range(300):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key='timeout')
        if event == 'Cancel' or event is None:
            break
        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.update_bar(i + 1)
    # done with loop... need to destroy the window as it's still open
    window.CloseNonBlocking()


def seq(start, end, step):
    if step == 0:
        raise ValueError("step must not be 0")
    sample_count = int(abs(end - start) / step)
    return itertools.islice(itertools.count(start, step), sample_count)


def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


def make_window(theme):
    sg.theme(theme)
    menu_def = [['&Application', ['E&xit']],
                ['&Help', ['&About']]]
    right_click_menu_def = [[], ['Edit Me', 'Versions', 'Nothing', 'More Nothing', 'Exit']]

    input_layout = [
        [sg.Menu(menu_def, key='-MENU-')],
        [sg.Text('Anything that requires user-input is in this tab!')],
        [sg.Text('sol_per_population', size=(20, 1)),
         sg.Spin(values=[i for i in range(1, 1000)], initial_value=30, size=(6, 1))],
        [sg.Text('mutation_probability', size=(20, 1), pad=((5, 3))),
         sg.Spin(values=[round(i, 2) for i in seq(0, 1.01, 0.01)], initial_value=0.5, size=(6, 1))],
        [sg.Text('num_generation', size=(20, 1)),
         sg.Spin(values=[i for i in range(1, 1000)], initial_value=30, size=(6, 1))],
        [sg.Text('crossover_probability', size=(20, 1), pad=((5, 3))),
         sg.Spin(values=[round(i, 2) for i in seq(0, 1.01, 0.01)], initial_value=0.5, size=(6, 1))],
        [sg.Text('num_parents_mating', size=(20, 1)), sg.Input(default_text='7', size=(8, 1))],
        [sg.Text('crossover_type', size=(20, 1)),
         sg.Drop(values=('single_point', 'two_points', 'uniform', 'scattered'))],
        [sg.Text('parent_selection_type', size=(20, 1)),
         sg.Drop(values=('sss', 'rws', 'sus', 'rank', 'random', 'tournament'))],
        [sg.Text('mutation_type', size=(20, 1)),
         sg.Drop(values=('random', 'swap', 'inversion', 'scramble', 'adaptive'))],
        [sg.Text('keep_parents', size=(20, 1)),
         sg.Spin(values=[i for i in range(-1, 100)], initial_value=-1, size=(6, 1))],
        [sg.Text('mutation_by_replacement', size=(20, 1)), sg.Drop(values=('False', 'True'))],
        [sg.Text('K_tournament', size=(20, 1)),
         sg.Spin(values=[i for i in range(1, 100)], initial_value=3, size=(6, 1))],
        [sg.Text('random_mutation_min_val', size=(20, 1)), sg.Input(default_text='-1.0', size=(8, 1))],
        [sg.Text('random_mutation_max_val', size=(20, 1)), sg.Input(default_text='1.0', size=(8, 1))],
        [sg.Text('stop_criteria', size=(20, 1)), sg.Drop(values=('None', 'reach_1000', 'saturate_15'))],
        [sg.Button("SUBMIT"), sg.Button("RUN"), sg.Button('Exit')]
    ]

    logging_layout = [
        [sg.Text("Anything printed will display here!")],
        [sg.Multiline(size=(60, 15), font='Courier 8', expand_x=True, expand_y=True, write_only=True,
                      reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True, autoscroll=True,
                      auto_refresh=True)]
    ]

    plot_1_layout = [
        [sg.T('Graph: Bouc-Wen')],
        [sg.B('Exit')],
        [sg.T('Controls:')],
        [sg.Canvas(key='controls_cv_1')],
        [sg.T('Figure:')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv_1',
                           # it's important that you set this size
                           size=(600 * 2, 600)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )]
    ]

    plot_2_layout = [
        [sg.T('Graph: Bouc-Wen')],
        [sg.B('Exit')],
        [sg.T('Controls:')],
        [sg.Canvas(key='controls_cv_2')],
        [sg.T('Figure:')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv_2',
                           # it's important that you set this size
                           size=(600 * 2, 600)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )]
    ]

    popup_layout = [
        [sg.Text("Load data")],
        [sg.Button("Open File"), sg.Button("Clear")],
        [sg.Multiline(size=(60, 15), font='Courier 8', expand_x=True, expand_y=True, write_only=True,
                      reroute_stdout=False, reroute_stderr=False, echo_stdout_stderr=True, autoscroll=True,
                      auto_refresh=True, key='-INPUT_SHOW-')]
    ]

    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              [sg.Text('Insert inputs, run program and display your results!', size=(100, 1), justification='center',
                       font=("Helvetica", 16),
                       relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]

    layout += [[sg.TabGroup([[sg.Tab('Input Elements', input_layout),
                              sg.Tab('Popups', popup_layout),
                              sg.Tab('Plot_1', plot_1_layout),
                              sg.Tab('Plot_2', plot_2_layout),
                              sg.Tab('Output', logging_layout)]], key='-TAB GROUP-', expand_x=True, expand_y=True)]]

    layout[-1].append(sg.Sizegrip())

    window = sg.Window('Genetic algorithm', layout, right_click_menu=right_click_menu_def,
                       right_click_menu_tearoff=True, grab_anywhere=False, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=False,
                       # scaling=2.0,
                       )
    window.set_min_size(window.size)
    return window


def main():
    window = make_window(sg.theme())
    desired_output, function_inputs = [], []
    # This is an Event Loop
    while True:
        event, values = window.read(timeout=100)
        # keep an animation running so show things are happening
        if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
            print('============ Event = ', event, ' ==============')
        if event in (None, 'Exit', 'Exit1', 'Exit2', 'Exit3'):
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
        elif event == "Open File":
            print("[LOG] Clicked Open File!")
            folder_or_file = sg.popup_get_file('Choose your file', keep_on_top=True)
            sg.popup("You chose: " + str(folder_or_file), keep_on_top=True)
            df = pd.read_csv(folder_or_file, encoding='unicode_escape')
            df[['Sila', 'Przemieszczenie']] = df['Sila;Przemieszczenie'].str.split(';', expand=True)
            df['Sila'] = pd.to_numeric(df['Sila'])
            df['Przemieszczenie'] = pd.to_numeric(df['Przemieszczenie'])
            desired_output = df['Sila'].to_list()
            function_inputs = df['Przemieszczenie'].to_list()
            window['-INPUT_SHOW-'].print('desired output FORCE: ', 2 * '\n', desired_output, '\n',
                                         'Data type: ' + str(type(desired_output[0])), 2 * '\n',
                                         'function inputs DISPLACEMENTS: ', 2 * '\n', function_inputs, '\n',
                                         'Data type: ' + str(type(function_inputs[0])))
            print("[LOG] User chose file: " + str(folder_or_file))
        elif event == "Clear":
            print("[LOG] Clear loaded data!")
            window['-INPUT_SHOW-'].update('')
            desired_output, function_inputs = [], []
        elif event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Versions':
            sg.popup(sg.get_versions(), keep_on_top=True)
        elif event == 'SUBMIT':
            print(values[5])
            print('-------- Values Dictionary (key=value) --------')
            for key in values:
                print(key, ' = ', values[key])
        elif event == 'RUN':
            if desired_output == [] and function_inputs == []:
                print("[LOG] You can't RUN program without loading data!")
            else:
                print("[LOG] Data is loaded and program will be execute!")
                plt.ioff()
                if values[13] == "" or values[13] == 'None':
                    GA = GA_backend.GeneticOptimization(function_inputs, desired_output, values[0], values[1], values[2],
                                                   values[3], values[4], "%s" % values[5], "%s" % values[6],
                                                   "%s" % values[7], values[8], values[9], values[10],
                                                   values[11], values[12], None)
                else:
                    GA = GA_backend.GeneticOptimization(function_inputs, desired_output, values[0], values[1], values[2],
                                                   values[3], values[4], "%s" % values[5], "%s" % values[6],
                                                   "%s" % values[7], values[8], values[9], values[10],
                                                   values[11], values[12], "%s" % values[13])
                # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
                fig = GA[0]
                plt.figure(2)
                DPI = fig.get_dpi()
                # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
                fig.set_size_inches(600 * 2 / float(DPI), 600 / float(DPI))
                # ------------------------------- Instead of plt.show()
                draw_figure_w_toolbar(window['fig_cv_1'].TKCanvas, fig, window['controls_cv_1'].TKCanvas)
                print("[LOG] Draw a plot_1!")

                fig_2 = GA[1]
                plt.figure(2)
                DPI = fig_2.get_dpi()
                # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
                fig_2.set_size_inches(600 * 2 / float(DPI), 600 / float(DPI))
                # ------------------------------- Instead of plt.show()
                draw_figure_w_toolbar(window['fig_cv_2'].TKCanvas, fig_2, window['controls_cv_2'].TKCanvas)
                print("[LOG] Draw a plot_2!")
    window.close()
    exit(0)


if __name__ == '__main__':
    run()
