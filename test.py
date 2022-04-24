def test():
    return 'abc', 100

word = 50
print('"%s"' % word)


fig = GA[0]
                plt.figure(2)
                DPI = fig.get_dpi()
                fig.set_size_inches(600 * 2 / float(DPI), 600 / float(DPI))
                draw_figure_w_toolbar(window['fig_cv_1'].TKCanvas, fig, window['controls_cv_1'].TKCanvas)
                print("[LOG] Draw a plot_1!")

fig_2 = GA[1]
                plt.figure(2)
                DPI = fig_2.get_dpi()
                fig_2.set_size_inches(600 * 2 / float(DPI), 600 / float(DPI))
                draw_figure_w_toolbar(window['fig_cv_2'].TKCanvas, fig_2, window['controls_cv_2'].TKCanvas)
                print("[LOG] Draw a plot_2!")
