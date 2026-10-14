""" Computational model for salience coding in apical dendrites of L5 pyramidal neurons

This is the main script to run the simulations and plot the panels of the manuscript:
"Error representations in apical dendrites of neocortical layer 5 pyramidal neurons during learning" by G. Schoenfeld,
S. Kollmorgen, M.C. Tsai, C. Lewis, S. Han, P. Bethge, A.M. Reuss, A. Aguzzi, W. Senn, V. Mante and F. Helmchen.

Plotting requires the results of the simulations to be present in the 'results' directory. The results can be generated
by running the simulations with the command:
>> python main.py simulate
The simulations might take a while to run depending on the hardware. They end by extracting the performance traces with
a Python port of the method of Smith et al. (2004) and merging them into the results. The performance traces can be
extracted again with:
>> python main.py perf
Alternatively, the Matlab script 'getperfs.m' (from Smith et al., 2004) located in the 'Smith' directory extracts them
from the same trial outcomes, after which they have to be merged into the main results by running:
>> python main.py perf matlab
Finally, all the panels can be plotted by running:
>> python main.py plot
If no argument is provided, the script will default to plotting all the panels.
"""

import sys
import l5apical

__author__ = "Matthias Chinyen Tsai"
__license__ = "GPL"
__version__ = "2.0.0"

if __name__ == '__main__':

    if len(sys.argv) == 1:
        l5apical.panels.plot_all()
    else:
        if str(sys.argv[1]) == 'plot':
            l5apical.panels.plot_all()

        elif str(sys.argv[1]) == 'simulate':
            l5apical.simulations.run()

        elif str(sys.argv[1]) == 'perf':
            if len(sys.argv) < 3 or str(sys.argv[2]) != 'matlab':
                l5apical.simulations.compute_smith_perf()
            l5apical.simulations.load_smith_perf()

        # Explain correct usage syntax
        else:
            print("Proper usage requires argument 'plot' or 'simulate' or 'perf'. To plot all the panels enter: \n"
                  ">> python main.py plot\n"
                  "\nTo run all the simulations (including the extraction of the performance traces) enter:\n"
                  ">> python main.py simulate\n"
                  "\nTo extract the performance traces again enter:\n"
                  ">> python main.py perf\n"
                  "\nTo use the performance traces extracted by the Matlab script getperfs.m located in the Smith "
                  "directory instead, run it after the simulations and then enter:\n"
                  ">> python main.py perf matlab\n"
                  "\nIf a step is interrupted, you might corrupt the results and might not be able to plot the panels.")
