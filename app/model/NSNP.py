import json, random, time, numpy as np
from pprint import pprint
from copy import deepcopy

class NumericalSNPSystem:
	def __init__( self, system_description):
		self.neurons = system_description['neurons']
		self.synapses = system_description['syn']
		self.reg_neurons = self.get_reg_neurons()
		self.out_neurons = self.get_out_neurons()

		# ORDERING SYSTEM DETAILS
		self.get_nrn_order()
		self.get_out_order()
		self.get_var_order()
		self.get_prf_order()

		# INITIAL MATRICES
		self.get_init_cmatrix()
		self.get_fmatrix()
		self.get_lmatrix()

		# MAPPING DETAILS
		self.map_neuron_to_var()
		self.map_neuron_to_neuron()
		self.map_func_to_var()
		self.map_func_contains_var()
		self.map_func_to_out()

		# INITIAL STATE
		self.unexplored_states = self.config_mx.tolist()
		self.explored_states = []
		self.state_graph = {
			'out_ord' : self.output_keys,
			'nrn_ord' : self.neuron_keys,
			'nodes': {},
		}
		
		# CONFIGURATION DEPENDENT MATRICES
		# spiking_mx = self.get_smatrix(self.config_mx[0])
		# production_mx = self.get_pmatrix(self.config_mx[0])
		# variable_mx = self.get_vmatrix(spiking_mx,self.config_mx[0])
	
	#===========================================================================
	# Get the some details of the NSN P system
	#---------------------------------------------------------------------------
	# get_out_neurons() - Get the output neurons
	#===========================================================================
	def get_reg_neurons(self):
		return {neuron['id']:neuron['data']
		  			for neuron in self.neurons if neuron['data']['ntype'] == 'reg'}
	
	def get_out_neurons(self):
		return {neuron['id']:0
		  			for neuron in self.neurons if neuron['data']['ntype'] == 'out'}

	#===========================================================================
	# Get the order of the variables and functions
	#---------------------------------------------------------------------------
	# get_var_order() - Variables are ordered by the order of the neurons
	# get_prf_order() - Functions are ordered by the order of the neurons
	# get_nrn_order() - Neurons are ordered by the order of the neurons
	# get_out_order() - Output neurons are ordered by the order of the neurons
	#===========================================================================
	def get_var_order(self):
		self.variables = [var for neuron in self.reg_neurons
							for var in self.reg_neurons[neuron]['var_']]
		
		self.variable_keys = [var[0] for var in self.variables]

	def get_prf_order(self):
		self.functions = [prf for neuron in self.reg_neurons
							for prf in self.reg_neurons[neuron]['prf']]

		self.function_keys = [prf[0] for prf in self.functions]

	def get_nrn_order(self):
		self.neuron_keys = [neuron for neuron in self.reg_neurons]

	def get_out_order(self):
		self.output_keys = [neuron for neuron in self.out_neurons]

	#===========================================================================
	# Helper functions for building the Spiking Matrix
	#---------------------------------------------------------------------------
	# check_threshold() - Check if the a function satisfies the threshold
	# compute_active_functions() - Active functions clusterred to each neuron
	# get_active_functions() - Get the indices of active functions in a neuron
	#===========================================================================
	def check_threshold(self,config,function):
		function_def = self.functions[function]

		return all(config[var] >= function_def[1]
							if function_def[1] != None else True
								for var in self.var_to_func[function])

	def compute_active_functions(self,config,format=None):
		active = np.zeros((self.f_location_mx.shape[0], self.f_location_mx.shape[1]))
		active_count = np.zeros(self.f_location_mx.shape[1])

		for index_j, neuron in enumerate(self.f_location_mx.T):
			for index_i, function in enumerate(neuron):
				if self.check_threshold(config,index_i) and function:
					active[index_i, index_j] = 1
					active_count[index_j] += 1
				
		if format == 'array':
			return (active_count, [1 if 1 in function else 0 for function in active])
		
		return (active_count, active)
		

	def get_active_functions(self,neuron,active):
		active = active.T[neuron]
		indices = [index for index, function in enumerate(active) if function]
		return indices

	#===========================================================================
	# Helper functions for building the Production Matrix
	#---------------------------------------------------------------------------
	# map_neuron_to_var() - Get the mapping of neurons to variables
	# map_neuron_to_neuron() - Get the mapping of neurons to neurons (synapses)
	# map_func_to_var() - Get the mapping of functions to variables (production)
	# map_func_contains_var() - Get the mapping of variables to functions (function uses variable)
	#===========================================================================
	def map_neuron_to_var(self):
		self.neuron_to_var = dict(); 
		
		mapped = 0
		for index, neuron in enumerate(self.reg_neurons):
			self.neuron_to_var[index] = list(range(mapped, mapped+len(self.reg_neurons[neuron]['var_'])))
			mapped += len(self.reg_neurons[neuron]['var_'])

	def map_neuron_to_neuron(self):
		self.neuron_to_neuron = {index:[] for index in range(len(self.reg_neurons))}
		self.neuron_to_out = {index:[] for index in range(len(self.reg_neurons))}
		
		for synapse in self.synapses:
			if synapse['target'] in self.out_neurons:
				source = self.neuron_keys.index(synapse['source'])
				self.neuron_to_out[source] += [synapse['target']]
			else:
				source = self.neuron_keys.index(synapse['source'])
				target = self.neuron_keys.index(synapse['target'])
				self.neuron_to_neuron[source].append(target)

	def map_func_to_var(self):
		self.func_to_var = {index:[] for index in range(len(self.functions))}

		for index, prf in enumerate(self.f_location_mx):
			belongs_to = np.where(np.isclose(prf, 1))[0][0]
			for neuron in self.neuron_to_neuron[belongs_to]:
				self.func_to_var[index] += self.neuron_to_var[neuron]

	def map_func_contains_var(self):
		self.var_to_func = {index:[] for index in range(len(self.functions))}
		
		mapped = 0
		for index, neuron in enumerate(self.reg_neurons):
			for index_j, prf in enumerate(self.reg_neurons[neuron]['prf']):
				for index_k, var in enumerate(self.neuron_to_var[index]):
					if prf[2][index_k][1] != 0:
						self.var_to_func[index_j+mapped].append(var)
			mapped += len(self.reg_neurons[neuron]['prf'])

	def map_func_to_out(self):
		self.func_to_out = {index:[] for index in range(len(self.functions))}
		
		mapped = 0
		for index, neuron in enumerate(self.reg_neurons):
			for index_j, var in enumerate(self.reg_neurons[neuron]['prf']):
				self.func_to_out[index_j+mapped] += self.neuron_to_out[index]
			mapped += len(self.reg_neurons[neuron]['prf'])

	#===========================================================================
	# Get the matrices for the NSN P system
	#---------------------------------------------------------------------------
	# get_init_cmatrix() - initial Configuration matrix
	# get_fmatrix() - Function matrix
	# get_lmatrix() - Function Location matrix
	# --------------------------------------------------------------------------
	# get_smatrix() - Spiking matrix
	# get_pmatrix() - Production matrix and Environment matrix
	# get_vmatrix() - Variable matrix
	#===========================================================================
	def get_init_cmatrix(self):
		self.config_mx = np.array([[var[1]
									for neuron in self.neuron_keys
									for var in self.reg_neurons[neuron]['var_']
								]],dtype=float)

	def get_fmatrix(self):
		self.function_mx = np.zeros((len(self.functions), len(self.variables)))

		mapped_coefs = 0; mapped_funcs = 0
		for index_i, neuron in enumerate(self.reg_neurons):
			for index_j, function in enumerate(self.reg_neurons[neuron]['prf']):
				for index_k, variable in enumerate(function[2]):
					self.function_mx[index_j+mapped_funcs, index_k+mapped_coefs] =\
						function[2][index_k][1]
			mapped_coefs += len(self.reg_neurons[neuron]['var_'])
			mapped_funcs += len(self.reg_neurons[neuron]['prf'])

	def get_lmatrix(self):
		self.f_location_mx = np.zeros((len(self.functions), len(self.reg_neurons)))

		placed_funcs = 0
		for index_i, neuron in enumerate(self.reg_neurons):
			for index_j, function in enumerate(self.reg_neurons[neuron]['prf']):
				self.f_location_mx[index_j+placed_funcs, index_i] = 1
			placed_funcs += len(self.reg_neurons[neuron]['prf'])

	def get_smatrix(self,config,branch=None):
		if branch == 'initial':
			spike = self.compute_random_spike(config)
			return np.tile(np.array(spike, dtype=float), (1, 1))

		active_count, active = self.compute_active_functions(config)
		comb_count = np.prod(active_count, where = active_count > 0)
		spiking_mx = np.zeros((int(comb_count), len(self.functions)))
		temp_comb_count = comb_count
		
		for index_m, neuron in enumerate(self.reg_neurons):
			if active_count[index_m] == 0:
				continue
			
			functions = self.get_active_functions(index_m,active)
			amount = temp_comb_count/active_count[index_m]
			repeats = comb_count/temp_comb_count
			index_i = 0

			for index_k in range(int(repeats)):
				for index_j in functions:
					counter = 0
					while counter < amount:
						spiking_mx[index_i, index_j] = 1
						counter = counter + 1
						index_i = index_i + 1

			temp_comb_count /= active_count[index_m]

		if branch is None:
			branch = 1 if len(spiking_mx) > 1 else None
		if branch is not None:
			temp_branch = branch if branch < comb_count else int(comb_count)
			indices = np.random.choice(\
				spiking_mx.shape[0], temp_branch, replace=False)
			spiking_mx = spiking_mx[indices]
		
		return spiking_mx

	def get_pmatrix(self,config):
		production_mx = np.zeros((len(self.functions), len(self.variables)))
		environment_mx = np.zeros((len(self.functions), len(self.out_neurons)))

		for index_i, function in enumerate(self.function_mx):
			sum = 0
			for index_j, coefficient in enumerate(function):
				sum += coefficient * config[index_j]

			for index_j in self.func_to_var[index_i]:
				production_mx[index_i, index_j] = sum

			for index_j in self.func_to_out[index_i]:
				environment_mx[index_i, self.output_keys.index(index_j)] = sum
		
		return production_mx, environment_mx

	def get_vmatrix(self,spiking_mx,config):
		variable_mx = np.array([config for i in range(spiking_mx.shape[0])])

		for index_i, row in enumerate(spiking_mx):
			for index_k, function in enumerate(row):
				if function == 1:
					for variable in self.var_to_func[index_k]:
						variable_mx[index_i][variable] = 0
		
		return variable_mx
		
	#===========================================================================
	# Main algorithm for simulating NSN P Systems
	#---------------------------------------------------------------------------
	# simulate() - Simulate the NSN P system up to a certain depth
	# simulate_single() - Simulate a single configuration
	#===========================================================================
	def simulate(self,branch=None,cur_depth=0,sim_depth=1):
		depth = cur_depth
		
		while depth < sim_depth:

			if not self.unexplored_states:
				break

			for config in deepcopy(self.unexplored_states):
				self.simulate_single(config, branch)
			
			depth = depth + 1

	def simulate_single(self,config,branch=None,spike=None):
		next_states = []

		S = self.get_smatrix(np.array(config),branch) if spike is None else\
			np.tile(np.array(spike, dtype=float), (1, 1))
		P, E = self.get_pmatrix(np.array(config))
		V = self.get_vmatrix(S,np.array(config))
		ENG = np.matmul(S,E)
		NG = np.matmul(S,P)
		C = np.add(NG,V)

		for state in C.tolist():
					if state not in self.explored_states \
											+ self.unexplored_states + next_states:
						next_states.append(state)

		self.state_graph['nodes'][tuple(config)] = {
			'next': C.tolist(), # list(k for k,_ in itertools.groupby(C.tolist()))
			'spike': S.tolist(),
			'env': ENG.tolist(),
			'matrices': {
				'P': P.tolist(),
				'E': E.tolist(),
				'V': V.tolist(),
			}
		}

		if config not in self.explored_states:
			self.explored_states.append(config)
			self.unexplored_states.remove(config)
			self.unexplored_states += next_states
	
	#===========================================================================
	# Helper functions for formatting the output
	#---------------------------------------------------------------------------
	# format_config() - Format the configuration
	# get_config_details() - Get the details of the configuration
	#---------------------------------------------------------------------------
	def format_config(self, node, index):
		return {
			'next': node['next'][index],
			'spike': node['spike'][index],
			'env': node['env'][index]
		}
	
	def get_config_details(self,config):
		key = tuple(config)
		node = self.state_graph['nodes'][key]
		return {
			'next' : node['next'],
			'spike' : node['spike'],
			'env' : node['env']
		}
	
	def get_config_matrices(self,config):
		key = tuple(config)
		node = self.state_graph['nodes'][key]
		matrices = deepcopy(node['matrices'])
		matrices['S'] = node['spike']
		matrices['F'] = self.function_mx.tolist()
		matrices['L'] = self.f_location_mx.tolist()
		return matrices
	
	#===========================================================================
	# Configuration manager (used by the API)
	#---------------------------------------------------------------------------
	# next() - Get the next configuration
	# add_spike() - Add a spike to the node
	# compute_random_spike() - Compute a random spike
	# get_active_array() - Get the active array
	#===========================================================================
	def add_spike(self, config, node, spike):
		S = np.tile(np.array(spike, dtype=float), (1, 1))
		P, E = self.get_pmatrix(np.array(config))
		V = self.get_vmatrix(S,np.array(config))
		ENG = np.matmul(S,E)
		NG = np.matmul(S,P)
		C = np.add(NG,V)

		next_states = []
		for state in C.tolist():
					if state not in self.explored_states \
											+ self.unexplored_states + next_states:
						next_states.append(state)

		node['next'] += C.tolist()
		node['spike'] += S.tolist()
		node['env'] += ENG.tolist()

		self.unexplored_states += next_states

	def compute_random_spike(self,config):
		active_count, active = self.compute_active_functions(config)
		
		# if config in self.explored_states:
		# 	key = tuple(config)
		# 	comb_count = np.prod(active_count, where = active_count > 0)
		# 	spike_count = len(self.state_graph['nodes'][key]['spike'])
		# 	if(comb_count == spike_count):
		# 		return self.state_graph['nodes'][key]['spike'][
		# 			random.randint(0,comb_count-1)
		# 		]

		spike = np.zeros(len(self.functions), dtype=int)
		for index_m in range(len(self.reg_neurons)):
			if active_count[index_m] == 0:
				continue
			
			functions = self.get_active_functions(index_m,active)
			index_n = random.choice(functions)
			spike[index_n] = 1

		return spike.tolist()

	def next(self, config, spike=None):
		key = tuple(config)
		
		if spike is None:
			spike = self.compute_random_spike(config)
		
		if config not in self.explored_states:
			self.simulate_single(config, 'initial', spike)

		node = self.state_graph['nodes'][key]
		
		if spike not in node['spike']:
			self.add_spike(config, node, spike)
		
		index = node['spike'].index(spike)
		return self.format_config(node, index)
	
	def get_active_array(self,config):
		active_count, active = self.compute_active_functions(config,format='array')
		return active
	
	def get_state_graph(self):
		formatted = deepcopy(self.state_graph)
		formatted['nodes'] = [
			{'conf' : list(k), 'next' : v['next'], 'spike' : v['spike'], 'env' : v['env']}
			for k,v in formatted['nodes'].items()
		]

		return formatted
	
	

# ACTIVE FUNCTIONS MATRIX EXAMPLE
# 		1 | 0 | 0
# 		1 | 0 | 0
# 		0 | 1 | 0
# 		0 | 1 | 0
# 		0 | 0 | 1
# 		0 | 0 | 1
# SPIKING MATRIX EXAMPLE
# 		1 0 1 0 1 0
# 		1 0 1 0 0 1
# 		1 0 0 1 1 0
# 		1 0 0 1 0 1
# 		0 1 1 0 1 0
# 		0 1 1 0 0 1
# 		0 1 0 1 1 0
# 		0 1 0 1 0 1

if __name__ == '__main__':
	import os
	import sys

	current_dir = os.path.dirname(os.path.abspath(__file__))
	parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
	parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
	sys.path.append(parent_dir)

	from app.middleware.nsnp_validation import NSNPSchema

	with open('app/tests/chain/all-chain-500-loop.json', 'r') as f:
		data = json.load(f)

	schema = NSNPSchema()
	system = NumericalSNPSystem(
        schema.load({
            'neurons' : data['nodes'],
            'syn' : data['edges']
        })
    )

	# Initial simulation
	start = time.time()
	system.simulate(branch='initial')
	end = time.time()

	elapsed_time = end - start
	print(elapsed_time)

	# Get next config
	state_graph = system.get_state_graph()
	initial_config = state_graph['nodes'][0]
	next_config = initial_config['next'][0]
	
	for i in range(5):
		start = time.time()
		config_details = system.next(next_config, None)
		end = time.time()

		next_config = config_details['next']
		elapsed_time = end - start
		print(elapsed_time)