import numpy as np
import matplotlib.pyplot as plt
import itertools

def parameter_sweep(sim_func, repeats=1, **params):
	""" Performs a parameter sweep of a simulation function with certain parameters.
	Parameters are passed in the 'params' section as either a fixed value or a list
	of values. Each parameter combination is simulated with sim_func.
	The results of the sweep are passed out as a numpy multidimensional
	array with dimensions (# params... , repeats, # state variables, time). 

	Arguments:
	----------
	- sim_func: simulation function. Takes in given
	parameters as keyword arguments and returns time
	series for simulation-related state variables
	as 1d numpy arrays.
	- repeat: number of repeats per param set (default 1)
	- **params: keyword-specified parameters for
	simulation. For each named param, value is either
	constant (not swept) or a list of values to be swept
	through.

	Returns:
	--------
	- results: numpy multidimensional array with dimensions (# params... , repeats, # state variables, time).
	The first axes correspond to each parameter being swept (params with fixed value are not counted).

	"""

	sweep_dict = {k: v for (k, v) in params.items() if isinstance(v,list)} 
	fixed_dict = {k: v for (k, v) in params.items() if not isinstance(v, list)}
	names_list = list(sweep_dict.keys())
	values_list = list(sweep_dict.values())

	for paramset in itertools.product(*values_list):
		print(paramset)
		idxes = tuple(values_list[i].index(paramset[i]) for i in range(len(paramset)))
		args_dict = {names_list[i]: paramset[i] for i in range(len(paramset))}
		for i in range(repeats):
			print(i)
			simulation_results = sim_func(**fixed_dict, **args_dict)
			if 'results' not in vars():
				results = np.zeros(tuple(len(item) for item in values_list) + (repeats, len(simulation_results), len(simulation_results[0])))
			for j in range(len(simulation_results)):
				results[idxes][i,j,:] = simulation_results[j]

	return results

def markov_quarantine_sim(alpha=0.5, beta=0.5, kappa=0.06, gamma=0.125, N=720720, H=4, tmax=100, N_init=20, time_factor=10, return_all=True):
	""" Runs a simulation of an infection in a population divided into households. Unlike
	in the agent-based simulation, the baseline unit in this simulation is a household,
	rather than an individual.

	Arguments:
	----------
	- alpha: rate of household infection
	- beta: rate of panmictic infection
	- kappa: rate of transitioning to a symptomatic state from an asymptomatic state.
	- gamma: rate of recovering from an infected state.
	- N: total population in simulation.
	- H: size of household unit
	- tmax: number of iterations.
	- N_init: number of initially infected individuals.
	- time_factor: number of simulation steps per day. Using a larger time_factor
	can be helpful when alpha >> beta.
	- return_all: returns 

	Returns:
	--------
	- s_counts: number of susceptible households at each time
	- f_counts: number of households with one infection
	- fy_counts: number of quarantined households with one
	- g_counts: number of fully infected households
	- gy_counts: number of quarantined fully infected households
	- r_counts: number of recovered households

	"""

	N = N - N%H
	""" states:
	- 0: uninfected household
	- 1: one infection, unquarantined
	- 2: one infection, quarantined
	- 3: fully infected, unquarantined
	- 4: fully infected, quarantined
	- 5: everyone is recovered

	"""
	households = np.zeros(N//H)
	households[:N_init] = 1

	s_counts = [N//H - N_init]
	f_counts = [N_init]
	fy_counts = [0]
	g_counts = [0]
	gy_counts = [0]
	r_counts = [0]

	for i in range(tmax):
		# infection phase
		randomness = np.random.rand(N//H)
		households += (np.logical_and(randomness > (1 - beta*H/N/time_factor)**(f_counts[-1]+H*g_counts[-1]), households==0)
			+ 2*np.logical_and(randomness < alpha/time_factor, np.logical_or(households==1, households==2)))

		# symptoms phase
		randomness = np.random.rand(N//H)
		households += np.logical_and(randomness < kappa/time_factor, households==1) + np.logical_and(randomness < H*kappa/time_factor, households==3)

		# recovery phase
		randomness = np.random.rand(N//H)
		households += (5-households)*np.logical_and(randomness < gamma/time_factor, households>0)

		s_counts.append(np.count_nonzero(households==0))
		f_counts.append(np.count_nonzero(households==1))
		fy_counts.append(np.count_nonzero(households==2))
		g_counts.append(np.count_nonzero(households==3))
		gy_counts.append(np.count_nonzero(households==4))
		r_counts.append(np.count_nonzero(households==5))

	return s_counts, f_counts, fy_counts, g_counts, gy_counts, r_counts

def agentbased_quarantine_sim(alpha=0.5, beta=0.5, kappa=0.06, gamma=0.125, N=720720, H=4, tmax=100, N_init=20, time_factor=10, return_all=True):
	""" Runs an agent-based simulation of an infection in a population divided into households.
	At each timestep, there are two phases of infection:
		- 1) panmictic ('in the streets'): every infected individual has a probability
		p0 of infecting any non-infected individual.
		- 2) household: people return to their houses. Every infected individual in
		each household (excluding those added in the last panmictic phase) can infect
		any non-infected individual in their house with probability ph.
	
	p0 is computed using the formula beta/N. Furthermore, infected individuals recover
	at a rate gamma.
	
	In addition to the above mechanics, each individual is initialized as an asymptomatic
	carrier, and becomes symptomatic with probability ps. In addition, there is an optional
	delay between the onset of infectivity and the exhibition of symptoms (this simulation does not distinguish
	between exposed but uninfectious and uninfected individuals). After a single member
	of a household shows symptoms, the entire household is quarantined, and no members
	can participate in the panmictic phase of the infection. The infection can be initialized
	with a large number of infected individuals in order to ensure that the infection
	does not peter out stochastically in its early phase.

	Parameters:
	-----------
	- alpha: rate of household infection
	- beta: rate of panmictic infection
	- kappa: rate of transitioning to a symptomatic state from an asymptomatic state.
	- gamma: rate of recovering from an infected state.
	- N: total population in simulation.
	- H: size of household unit
	- tmax: number of iterations.
	- N_init: number of initially infected individuals.
	- time_factor: number of simulation steps per day. Using a larger time_factor
	can be helpful when alpha >> beta.

	Returns:
	--------
	if return_all:
	- susceptible_totals: number of susceptible individuals at teach time step
	- infected_totals: number of infected at each time step of the simulation
	- recovered_totals: number of recovered at each time step of the simulation
	- active_infected_totals: number of non-quarantined infected at each time step

	
	"""
	
	# total number of people should be divisible by household size
	N = N - N%H
	# assert N % H == 0
	ph = alpha/time_factor
	p0 = (beta/N)/time_factor
	ps = kappa/time_factor
	pr = gamma/time_factor
	
	# pop = bitmap for the symptom state of each individual (-1 = uninfected, 0 = asymptomatic, 1 = symptomatic, 2 = recovered)
	pop = -np.ones(N)

	# tracks the number of infected individuals by household
	house_infected = np.zeros(N//H)
	# tracks whether a house is quarantined
	house_quarantined = np.zeros(N//H)
	
	# initialize infection
	init_idxes = np.random.choice(N, size=N_init)
	for idx in init_idxes:
		pop[idx] = 0
		house_infected[idx//H] += 1
	
	active_infected = np.count_nonzero(pop == 0)  # number of active participants in panmictic phase
	
	infected_totals = [active_infected]
	active_infected_totals = [active_infected]
	recovered_totals = [0]
	susceptible_totals = [N - active_infected]

	
	for i in range(tmax):
		# panmictic phase
		pop += np.logical_and(np.logical_and(pop==-1, (np.random.rand(N) > (1 - p0)**active_infected)), 1-house_quarantined[np.arange(N)//H])

		# household phase 
		household_probabilities = (1 - ph)**house_infected[np.arange(N) // H]
		pop += np.logical_and(pop == -1, (np.random.rand(N) > household_probabilities))

		# adjust quarantine status
		house_quarantined = np.any(pop.reshape(N//H, H) == 1, axis=1)

		# adjust symptoms
		pop += np.logical_and(pop == 0, np.random.rand(N) < ps)

		infecteds = np.logical_or(pop==0, pop==1)

		# adjust recovery
		pop += np.logical_and(infecteds, np.random.rand(N) < pr)

		# adjust counts
		house_infected = np.sum(infecteds.reshape(N//H, H),axis=1)
		active_infected = np.count_nonzero(np.logical_and(infecteds, 1-house_quarantined[np.arange(N)//H]))
		infected_totals.append(np.count_nonzero(infecteds))
		active_infected_totals.append(active_infected)
		recovered_totals.append(np.count_nonzero(pop == 2))
		susceptible_totals.append(N - recovered_totals[-1] - infected_totals[-1])

	if return_all:
		return susceptible_totals, infected_totals, recovered_totals, active_infected_totals

	return infected_totals

def agentbased_baseline_sim(alpha=0.5, beta=0.5, gamma=0.125, N=720720, H=4, tmax=100, N_init=20, time_factor=10, return_all=True):
	""" Runs an agent-based simulation of an infection in a population divided into households.
	At each timestep, there are two phases of infection:
		- 1) panmictic ('in the streets'): every infected individual has a probability
		p0 of infecting any non-infected individual.
		- 2) household: people return to their houses. Every infected individual in
		each household (excluding those added in the last panmictic phase) can infect
		any non-infected individual in their house with probability ph.
	
	p0 is computed using the formula beta/N/time_factor. Each infected individual
	recovers with a probability pr = gamma/time_factor.
	
	Parameters:
	-----------
	- alpha: household infection rate
	- beta: panmictic infection rate
	- gamma: rate of recovery
	- N: total population in simulation.
	- H: size of household unit
	- tmax: number of iterations.
	- N_init: number of initially infected individuals.
	- time_factor: number of simulation steps per day.
	A larger time_factor is helpful when alpha >> max(beta, 1)
		
	Returns:
	--------
	if return_all:
	- susceptible_totals: number of susceptible individuals at teach time step
	- infected_totals: number of infected at each time step of the simulation
	- recovered_totals: number of recovered at each time step of the simulation
	else:
	- infected_totals: number of infected at each time step of the simulation

	"""


	# total number of people should be divisible by household size
	# assert N % H == 0
	N = N - N%H

	p0 = (beta/N)/time_factor
	ph = alpha/time_factor
	pr = gamma/time_factor

	# pop = bitmap for each individual (-1 = susceptible, 0 = infected, 1 = recovered)
	pop = -np.ones(N)

	# tracks the number of infected individuals by household
	house_infected = np.zeros(N//H)
	
	# initialize infection
	init_idxes = np.random.choice(N, size=N_init)
	for idx in init_idxes:
		pop[idx] = 0
		house_infected[idx//H] += 1

	infected = N_init
	
	infected_totals = [N_init]
	recovered_totals = [0]
	susceptible_totals = [N-N_init]
	
	for i in range(tmax):
		# panmictic phase
		pop += np.logical_and(pop == -1, (np.random.rand(N) > (1 - p0)**infected))
		
		# household phase
		household_probabilities = (1 - ph)**house_infected[np.arange(N) // H]
		pop += np.logical_and(pop == -1, (np.random.rand(N) > household_probabilities))
		
		# recoveries
		pop += np.logical_and(pop == 0, np.random.rand(N) < pr)

		# adjust counts
		house_infected = np.sum((pop == 0).reshape(N//H, H),axis=1)
		infected = np.count_nonzero(pop == 0)
		infected_totals.append(infected)

		recovered_totals.append(np.count_nonzero(pop == 1))
		susceptible_totals.append(N-infected_totals[-1]-recovered_totals[-1])

	if return_all:
		return susceptible_totals, infected_totals, recovered_totals

	return infected_totals
