import numpy as np
from utils import *

np.save('./data/agentbased_baseline_parameter_sweep.npy', parameter_sweep(agentbased_baseline_sim,
																repeats=2,
																beta=list(np.linspace(0.05, 0.6, num=30)),
																alpha=list(np.linspace(0.6, 5, num=30)),
																H = [1,2,4,8,16],
																time_factor = 4,
																tmax = 200,
																gamma = 0.125,
																N = 200000))