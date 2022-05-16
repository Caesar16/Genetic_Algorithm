import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import hashlib
import itertools
import app_backend_poly as ABP
import app_backend_sinus as ABS

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
    layout = [[sg.Text('Initializing')],
              [sg.ProgressBar(300, orientation='h',
                              size=(20, 20), key='progress')],
              [sg.Cancel()]]
    window = sg.Window('Custom Progress Meter', layout)
    progress_bar = window['progress']
    for i in range(300):
        event, values = window.read(timeout=0, timeout_key='timeout')
        if event == 'Cancel' or event is None:
            break
        progress_bar.update_bar(i + 1)
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
        [sg.Radio('GA_sinus', "RadioDemo", default=True, size=(15, 1), k='-R1-'),
         sg.Radio('GA_polynomial', "RadioDemo", default=True, size=(15, 1), k='-R2-')],
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
        [sg.T('Graph:')],
        [sg.B('Exit')],
        [sg.T('Controls:')],
        [sg.Canvas(key='controls_cv_1')],
        [sg.T('Figure:')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv_1',
                           size=(600 * 2, 600)
                           )]
            ],
            background_color='#DAE0E6',
            pad=(0, 0)
        )]
    ]

    plot_2_layout = [
        [sg.T('Graph:')],
        [sg.B('Exit')],
        [sg.T('Controls:')],
        [sg.Canvas(key='controls_cv_2')],
        [sg.T('Figure:')],
        [sg.Column(
            layout=[
                [sg.Canvas(key='fig_cv_2',
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
                       use_custom_titlebar=True, finalize=True, keep_on_top=True,
                       # scaling=2.0,
                       )
    window.set_min_size(window.size)
    return window


def draw(figure, fig_cv, controls_cv, window):
    fig = figure
    plt.figure(2)
    DPI = fig.get_dpi()
    fig.set_size_inches(600 * 2 / float(DPI), 600 / float(DPI))
    draw_figure_w_toolbar(window[fig_cv].TKCanvas, fig, window[controls_cv].TKCanvas)

def main():
    window = make_window(sg.theme())
    desired_output, function_inputs = [], []
    while True:
        event, values = window.read(timeout=100)
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
            with open(folder_or_file) as f:
                for line in f:
                    function_inputs.append(float(line.split()[0]))
                    desired_output.append(float(line.split()[1]))
            window['-INPUT_SHOW-'].print('desired output: ', 2 * '\n', desired_output, '\n',
                                         'Data type: ' + str(type(desired_output[0])), 2 * '\n',
                                         'function inputs: ', 2 * '\n', function_inputs, '\n',
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
        elif event == 'RUN' and values['-R1-'] is False and values['-R2-'] is True:
            if desired_output == [] and function_inputs == []:
                print("[LOG] You can't RUN program without loading data!")
            else:
                print("[LOG] Data is loaded and program will be execute!")
                plt.ioff()
                if values[13] == "" or values[13] == 'None':
                    GA = ABP.GeneticOptimization(function_inputs, desired_output, values[0], values[1], values[2],
                                                   values[3], values[4], "%s" % values[5], "%s" % values[6],
                                                   "%s" % values[7], values[8], values[9], values[10],
                                                   values[11], values[12], None)
                else:
                    GA = ABP.GeneticOptimization(function_inputs, desired_output, values[0], values[1], values[2],
                                                   values[3], values[4], "%s" % values[5], "%s" % values[6],
                                                   "%s" % values[7], values[8], values[9], values[10],
                                                   values[11], values[12], "%s" % values[13])

                draw(GA[0], 'fig_cv_1', 'controls_cv_1', window)
                draw(GA[1], 'fig_cv_2', 'controls_cv_2', window)

        elif event == 'RUN' and values['-R2-'] is False and values['-R1-'] is True:
            if desired_output == [] and function_inputs == []:
                print("[LOG] You can't RUN program without loading data!")
            else:
                print("[LOG] Data is loaded and program will be execute!")
                plt.ioff()
                if values[13] == "" or values[13] == 'None':
                    GA = ABS.GeneticOptimization(function_inputs, desired_output, values[0], values[1], values[2],
                                                   values[3], values[4], "%s" % values[5], "%s" % values[6],
                                                   "%s" % values[7], values[8], values[9], values[10],
                                                   values[11], values[12], None)
                else:
                    GA = ABS.GeneticOptimization(function_inputs, desired_output, values[0], values[1], values[2],
                                                   values[3], values[4], "%s" % values[5], "%s" % values[6],
                                                   "%s" % values[7], values[8], values[9], values[10],
                                                   values[11], values[12], "%s" % values[13])

                draw(GA[0], 'fig_cv_1', 'controls_cv_1', window)
                draw(GA[1], 'fig_cv_2', 'controls_cv_2', window)
    window.close()
    exit(0)


if __name__ == '__main__':
    run()
