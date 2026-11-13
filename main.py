""" Computational model for salience coding in apical dendrites of L5 pyramidal neurons

This is the main script to run the simulations and plot the panels of the manuscript:
"Error representations in apical dendrites of neocortical layer 5 pyramidal neurons during learning" by G. Schoenfeld,
S. Kollmorgen, M.C. Tsai, C. Lewis, S. Han, P. Bethge, A.M. Reuss, A. Aguzzi, W. Senn, V. Mante and F. Helmchen.

Without an argument, the script runs all the simulations, including the extraction of the performance traces with the
method of Smith et al. (2004), saves their results in the 'results' directory and then plots all the panels:
>> python main.py
The simulations might take a while to run depending on the hardware. Each step can also be run on its own: the
simulations (including the performance traces) with
>> python main.py simulate
the extraction of the performance traces of the saved results again with
>> python main.py perf
and the plotting of all the panels from the saved results with
>> python main.py plot
"""

import sys
import l5apical

__author__ = "Matthias Chinyen Tsai"
__license__ = "GPL"
__version__ = "2.0.0"

if __name__ == '__main__':

    if len(sys.argv) == 1:
        l5apical.simulations.run()
        l5apical.panels.plot_all()
    else:
        if str(sys.argv[1]) == 'plot':
            l5apical.panels.plot_all()

        elif str(sys.argv[1]) == 'simulate':
            l5apical.simulations.run()

        elif str(sys.argv[1]) == 'perf':
            l5apical.simulations.extract_performances()

        # Explain correct usage syntax
        else:
            print("Proper usage requires no argument or the argument 'simulate', 'perf' or 'plot'. To run all the "
                  "simulations (including the extraction of the performance traces) and then plot all the panels "
                  "enter:\n"
                  ">> python main.py\n"
                  "\nTo only run all the simulations (including the extraction of the performance traces) enter:\n"
                  ">> python main.py simulate\n"
                  "\nTo extract the performance traces of the saved results again enter:\n"
                  ">> python main.py perf\n"
                  "\nTo only plot all the panels from the saved results enter:\n"
                  ">> python main.py plot")
