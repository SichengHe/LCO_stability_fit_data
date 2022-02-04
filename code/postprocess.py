from pyoptsparse import OPT, Optimization, History
import numpy as np

# hist = History("/home/hschsc/job_dir/LCO_stability_fit/code/snopt_LCO_ae_Hist.hst", flag="r")
hist = History("/home/hschsc/job_dir/LCO_stability_fit/code/snopt_LCO_Hist.hst", flag="r")
xdict = hist.getValues(names="xvars", major=True)

# filename = "opt_ae_hist.dat"
filename = "opt_hist.dat"
np.savetxt(filename, xdict["xvars"])

