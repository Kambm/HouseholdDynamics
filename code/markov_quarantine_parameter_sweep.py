import numpy as np
from utils import *

np.save('./data/markov_quarantine_parameter_sweep.npy', parameter_sweep(markov_quarantine_sim,
																repeats=5,
																beta=list(np.linspace(0.05, 0.6, num=20)),
																alpha=list(np.linspace(0.6, 5, num=20)),
																kappa=list(np.linspace(0, 0.6, num=20)),
																H = [1,2,4,8,16],
																time_factor = 10,
																tmax = 300,
																gamma = 0.125,
																N = 200000))