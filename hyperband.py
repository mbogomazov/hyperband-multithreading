import pickle, os.path, numpy as np
from random import random
from math import log, ceil
from time import time, ctime


class Hyperband:
	max_iter = 81  	# maximum iterations per configuration
	eta = 3			# defines configuration downsampling rate (default = 3)
	results = []	# list of dicts
	counter = 0
	best_loss = np.inf
	best_counter = -1

	def __init__(self, model):
		self.model = model
		self.logeta = lambda x: log(x) / log(self.eta)
		self.s_max = int(self.logeta( self.max_iter))
		self.B = (self.s_max + 1) * self.max_iter

	# Run each of the n configs for <iterations>
	# and keep best (n_configs / eta) configurations
	# function can be called multiple times
	def run(self, skip_last = 0, dry_run = False, backup_filename=''):

		T_backup = []
		results_backup = []
		best_loss_backup = np.inf
		best_counter_backup = 0
		early_stops_backup = []
		val_losses_backup = []
		s_backup = 0
		i_backup = 0
		t_ind_backup = 0
		counter_backup = 0

		if backup_filename and os.path.isfile(backup_filename):
			with open(backup_filename, 'rb') as f:
				last_state = pickle.load(f)
				T_backup = last_state['T']
				results_backup = last_state['results']
				best_loss_backup = last_state['best_loss']
				best_counter_backup = last_state['best_counter']
				early_stops_backup = last_state['early_stops']
				val_losses_backup = last_state['val_losses']
				s_backup = last_state['s']
				i_backup = last_state['i']
				t_ind_backup = last_state['t_ind']
				counter_backup = last_state['counter']
				self.counter = counter_backup
				self.results = results_backup
				self.best_counter = best_counter_backup
				self.best_loss = best_loss_backup

		for s in list(reversed(list(range(self.s_max + 1))))[self.s_max - s_backup:]:
			# initial number of configurations
			n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
			# initial number of iterations per config
			r = self.max_iter * self.eta ** -s
			# n random configurations
			T = [ self.model.get_params() for i in range(n)] if not T_backup else T_backup
			if T_backup:
				T_backup = []

			for i in range(i_backup, (s + 1) - int( skip_last)):
				n_configs = n * self.eta ** ( -i )
				n_iterations = r * self.eta ** ( i )

				print(("\n*** {} configurations x {:.1f} iterations each".format(
					n_configs, n_iterations )))

				val_losses = [] if not val_losses_backup else val_losses_backup
				early_stops = [] if not early_stops_backup else early_stops_backup
				if val_losses_backup:
					val_losses_backup = []
				if early_stops_backup:
					early_stops_backup = []

				for t_ind, t in enumerate(T):
					if t_ind < t_ind_backup:
						continue
					self.counter += 1
					print(("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
						self.counter, ctime(), self.best_loss, self.best_counter )))

					start_time = time()

					if dry_run:
						result = {'loss': random(), 'log_loss': random(), 'auc': random()}
					else:
						result = self.model.try_params( n_iterations, t )

					assert(type(result) == dict )
					assert('loss' in result)

					seconds = int(round(time() - start_time))
					print(("\n{} seconds.".format(seconds)))

					loss = result['loss']
					val_losses.append(loss)

					early_stop = result.get('early_stop', False)
					early_stops.append(early_stop)

					# keeping track of the best result so far (for display only)
					# could do it be checking results each time, but hey
					if loss < self.best_loss:
						self.best_loss = loss
						self.best_counter = self.counter

					result['counter'] = self.counter
					result['seconds'] = seconds
					result['params'] = t
					result['iterations'] = n_iterations

					if backup_filename:
						self.backup_data(backup_filename, T, early_stops, s, i, t_ind, val_losses)

				# select a number of best configurations for the next loop
				# filter out early stops, if any
				indices = np.argsort(val_losses)
				T = [T[i] for i in indices if not early_stops[i]]
				T = T[:int(n_configs / self.eta)]


		return self.results


	def backup_data(self, backup_filename, T, early_stops, s, i, t_ind, val_losses):
		cur_state = {
			'T': T,
			'results': self.results,
			'best_loss': self.best_loss,
			'best_counter': self.best_counter,
			'early_stops': early_stops,
			's': s,
			'i': i,
			't_ind': t_ind,
			'val_losses': val_losses,
			'counter': self.counter
		}
		with open(backup_filename, 'wb') as f:
			pickle.dump(cur_state, f)
			print('state backuped')
