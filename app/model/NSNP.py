import json, random, time, numpy as np
from pprint import pprint
from copy import deepcopy

class NumericalSNPSystem:
	def __init__( self, system_description):
		self.neurons = system_description['neurons']
		self.synapses = system_description['syn']
		self.reg_neurons = self.get_reg_neurons()
		self.out_neurons = self.get_out_neurons()
		self.in_neurons = self.get_in_neurons()

		# ORDERING SYSTEM DETAILS
		self.get_nrn_order()
		self.get_out_order()
		self.get_var_order()
		self.get_prf_order()
		self.get_in_order()

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
		self.map_in_to_var()

		# INITIAL STATE
		self.unexplored_states = self.config_mx.tolist()
		self.state_graph = {
			'out_ord' : self.output_keys,
			'nrn_ord' : self.neuron_keys,
			'in_ord' : self.input_keys,
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
		  			for neuron in self.neurons if neuron['data']['type'] == 'reg'}
	
	def get_out_neurons(self):
		return {neuron['id']:0
		  			for neuron in self.neurons if neuron['data']['type'] == 'out'}
	
	def get_in_neurons(self):
		return {neuron['id']:neuron['data']
		  			for neuron in self.neurons if neuron['data']['type'] == 'in'}

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

	def get_in_order(self):
		self.input_keys = [neuron for neuron in self.in_neurons]

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
		self.in_to_neuron = {index:[] for index in range(len(self.in_neurons))}
		
		for synapse in self.synapses:
			if synapse['target'] in self.out_neurons:
				source = self.neuron_keys.index(synapse['source'])
				target = self.output_keys.index(synapse['target'])
				self.neuron_to_out[source].append(target)
			elif synapse['source'] in self.in_neurons:
				source = self.input_keys.index(synapse['source'])
				target = self.neuron_keys.index(synapse['target'])
				self.in_to_neuron[source].append(target)
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

	def map_in_to_var(self):
		self.in_to_var = {index:[] for index in range(len(self.in_neurons))}

		for index, neuron in enumerate(self.in_neurons):
			for target in self.in_to_neuron[index]:
				self.in_to_var[index] += self.neuron_to_var[target]

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
			branch = 64 if len(spiking_mx) > 64 else None
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
				environment_mx[index_i, index_j] = sum
		
		return production_mx, environment_mx

	def get_vmatrix(self,spiking_mx,config):
		variable_mx = np.array([config for i in range(spiking_mx.shape[0])])

		for index_i, row in enumerate(spiking_mx):
			for index_k, function in enumerate(row):
				if function == 1:
					for variable in self.var_to_func[index_k]:
						variable_mx[index_i][variable] = 0
		
		return variable_mx
	
	def get_ivector(self,depth):
		input_vc = np.zeros((len(self.in_neurons)))
		for index_i, input_ in enumerate(self.input_keys):
			if depth < len(self.in_neurons[input_]['train']):
				input_vc[index_i] = self.in_neurons[input_]['train'][depth]
			# 	continue

			# input_vc[index_i] = 0

		return input_vc
	
	def get_imatrix(self,input_vc):
		input_mx = np.zeros((len(self.variables)))

		for index_i, neurons in enumerate(input_vc):
			for index_j, target in enumerate(self.in_to_var[index_i]):
				input_mx[target] += input_vc[index_i]

		return input_mx
	
	def get_inputs_details(self,depth):
		input_vc = self.get_ivector(depth)
		input_mx = self.get_imatrix(input_vc)

		return input_vc, input_mx
	
	#===========================================================================
	# Helper functions for simulating the NSN P system
	#---------------------------------------------------------------------------
	# should_explore() - Check if the configuration should be explored
	# add_to_state_graph() - Add/Edit the configuration to/in the state graph
	#---------------------------------------------------------------------------
	def should_explore(self, state, inputs, depth, next_states):
		f_state = (state + inputs).tolist()

		if tuple(f_state) in self.state_graph['nodes']:
			next_IV = tuple(self.get_ivector(depth + 1))
			if next_IV not in self.state_graph['nodes'][tuple(f_state)]['in']:
				next_states.append(f_state)

		elif f_state not in self.unexplored_states + next_states:
			next_states.append(f_state)

	def add_to_state_graph(self,config,S,P,E,V,IV,IM,ENG,CL):
		key = tuple(config)
		
		# If the configuration is already in the state graph
		if key in self.state_graph['nodes']:
			for index, spike in enumerate(S.tolist()):
				if spike not in self.state_graph['nodes'][key]['spike']:
					self.state_graph['nodes'][key]['spike'].append(spike)
					self.state_graph['nodes'][key]['next'].append(CL[index])
					self.state_graph['nodes'][key]['env'].append(ENG[index].tolist())
					self.state_graph['nodes'][key]['matrices']['V'].append(V[index].tolist())
			
			self.state_graph['nodes'][key]['in'][tuple(IV)] = {
				'state' : [
					self.state_graph['nodes'][key]['spike'].index(spike)
					for spike in S.tolist()
				],
				'input' : IM.tolist()
			}

		# If the configuration is not yet in the state graph
		else:
			self.state_graph['nodes'][key] = {
				'spike': S.tolist(),
				'env': ENG.tolist(),
				'next': CL,
				'in': {
					tuple(IV) : {
						'state' : [CL.index(state) for state in CL],
						'input' : IM,
					}
				},
				'matrices': {
					'P': P.tolist(),
					'E': E.tolist(),
					'V': V.tolist(),
				}
			}
		
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
				self.simulate_single(config, depth, branch)
			
			depth = depth + 1

	def simulate_single(self,config,depth,branch=None,spike=None):
		next_states = []

		S = self.get_smatrix(np.array(config),branch) if spike is None else\
			np.tile(np.array(spike, dtype=float), (1, 1))
		P, E = self.get_pmatrix(np.array(config))
		V = self.get_vmatrix(S,np.array(config))
		IV, IM = self.get_inputs_details(depth)
		ENG = np.matmul(S,E)
		NG = np.matmul(S,P)
		C = np.add(NG,V)
		CL = C.tolist()

		for state in C:
			self.should_explore(state, IM, depth, next_states)

		if config in self.unexplored_states:
			self.unexplored_states.remove(config)
		self.unexplored_states += next_states
		
		self.add_to_state_graph(config,S,P,E,V,IV,IM,ENG,CL)
		
	#===========================================================================
	# Helper functions for formatting the output
	#---------------------------------------------------------------------------
	# format_config() - Format the configuration
	# get_config_details() - Get the details of the configuration
	# get_config_matrices() - Get the matrices of the configuration
	#---------------------------------------------------------------------------
	def format_config(self, node, index, depth):
		next = node['next'][index]
		IV = self.get_ivector(depth)
		next = np.array(next) + node['in'][tuple(IV)]['input']

		return {
			'next': next.tolist(),
			'spike': node['spike'][index],
			'env': node['env'][index],
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
	# Configuration Manager (accessible by clients through API)
	#---------------------------------------------------------------------------
	# NOTE: The implementation of this only traverses one path at a time
	#---------------------------------------------------------------------------
	# next() - Get the next configuration
	# add_spike() - Add a spike to the node
	# compute_random_spike() - Compute a random spike
	# get_active_array() - Get the active array
	#===========================================================================
	def add_spike(self, config, depth, node, spike):
		next_states = []
		
		S = np.tile(np.array(spike, dtype=float), (1, 1))
		P, E = self.get_pmatrix(np.array(config))
		V = self.get_vmatrix(S,np.array(config))
		IV, IM = self.get_inputs_details(depth)
		ENG = np.matmul(S,E)
		NG = np.matmul(S,P)
		C = np.add(NG,V)
		CL = C.tolist()

		for state in C:
			self.should_explore(state, IM, depth, next_states)
		
		node['spike'] += S.tolist()
		node['env'] += ENG.tolist()
		node['next'] += CL
		
		if tuple(IV) not in node['in']:
			node['in'][tuple(IV)] = {
				'state' : [len(node['next'])-1],
				'input' : IM.tolist()
			}
		else:
			node['in'][tuple(IV)]['state'] += [
				len(node['next'])-1
			]
			
		node['matrices']['V'] += V.tolist()

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

	def next(self, config, depth, spike=None):
		key = tuple(config)
		
		if spike is None:
			spike = self.compute_random_spike(config)
		
		if key not in self.state_graph['nodes']:
			self.simulate_single(config, depth, 'initial', spike)

		node = self.state_graph['nodes'][key]
		
		if spike not in node['spike']:
			self.add_spike(config, depth, node, spike)
		
		index = node['spike'].index(spike)
		IV, IM = self.get_inputs_details(depth)

		if tuple(IV) not in node['in']:
			node['in'][tuple(IV)] = {
				'state' : [index],
				'input' : IM.tolist()
			}

		return self.format_config(node, index, depth)
	
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

	# with open('app/tests/custom/sample-from-ballesteros.json', 'r') as f:
	with open('app/tests/custom/input3.json', 'r') as f:
		data = json.load(f)

	schema = NSNPSchema()
	system = NumericalSNPSystem(
        schema.load({
            'neurons' : data['nodes'],
            'syn' : data['edges']
        })
    )

	system.simulate(sim_depth=10)
	pprint(system.state_graph)

	# # Initial simulation
	# start = time.time()
	# system.simulate(branch='initial')
	# end = time.time()

	# elapsed_time = end - start
	# print(elapsed_time)

	# # Get next config
	# state_graph = system.get_state_graph()
	# initial_config = state_graph['nodes'][0]
	# next_config = initial_config['next'][0]
	
	# for i in range(5):
	# 	start = time.time()
	# 	config_details = system.next(next_config, i, None)
	# 	end = time.time()

	# 	next_config = config_details['next']
	# 	elapsed_time = end - start
	# 	print(elapsed_time)