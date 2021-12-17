from IPython.display import clear_output
import pickle, os.path, numpy as np
from threading import Thread, Lock
from math import log, ceil
from time import sleep, time
from itertools import islice
from pprint import pprint


class Hyperband:
    max_iter = 81      # maximum iterations per configuration
    eta = 3            # defines configuration downsampling rate (default = 3)
    results = []    # list of dicts
    counter = 0
    best_loss = np.inf
    best_accuracy = 0
    best_counter = -1
    best_params = {}
    val_losses = []
    early_stops = []
    threads_num = 0
    finished_threads = 0

    def __init__(self, model):
        self.model = model
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta( self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter
        self.lock = Lock()

    # Run each of the n configs for <iterations>
    # and keep best (n_configs / eta) configurations
    # function can be called multiple times
    def run(self, backup_filename='', threads_num=5):
        self.threads_num = threads_num
        results_backup = []
        best_loss_backup = np.inf
        best_counter_backup = 0
        early_stops_backup = []
        val_losses_backup = []
        s_backup = None
        i_backup = None
        counter_backup = 0
        T_backup = []
        chunk_offset_backup = None
        finished_threads_backup = None

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
                chunk_offset_backup = last_state['chunk_offset']
                counter_backup = last_state['counter']
                finished_threads_backup = last_state['finished_threads_backup']
                self.counter = counter_backup
                self.results = results_backup
                self.best_counter = best_counter_backup
                self.best_loss = best_loss_backup
                self.best_accuracy = last_state['best_accuracy']
                self.best_params = last_state['best_params']
                print('Last session state restored!')
                sleep(2)

        for s in reversed(list(range(self.s_max + 1))):
            if s_backup != None and s_backup < s:
                continue
            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            # initial number of iterations per config
            r = self.max_iter * self.eta ** -s
            # n random configurations
            T = [self.model.get_params() for i in range(n)] if not T_backup else T_backup
            if T_backup:
                T_backup = []

            for i in range(s + 1):
                if i_backup != None and i < i_backup:
                    continue
                else:
                    i_backup = None

                self.n_configs = n * self.eta ** ( -i )
                self.n_iterations = r * self.eta ** ( i )

                self.val_losses = [] if not val_losses_backup else val_losses_backup
                self.early_stops = [] if not early_stops_backup else early_stops_backup
                if val_losses_backup:
                    val_losses_backup = []
                if early_stops_backup:
                    early_stops_backup = []

                chunks_list = self.gen_chunk(T, threads_num)
                for chunk_offset, ch in enumerate(chunks_list):
                    if chunk_offset_backup != None and chunk_offset < chunk_offset_backup:
                        continue
                    else:
                        chunk_offset_backup = None

                    threads = []
                    self.finished_threads = 0
                    self.print_progress(chunk_offset, len(chunks_list))
                    for task_offset, task in enumerate(ch):
                        if finished_threads_backup != None and task_offset < finished_threads_backup:
                            continue
                        else:
                            finished_threads_backup = None
                        t = Thread(target=self.try_params_in_thread, args=(task, backup_filename, s, i, T, chunk_offset, len(chunks_list)))
                        t.start()
                        threads.append(t)

                    for t in threads:
                        t.join()

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(self.val_losses)
                T = [T[i] for i in indices if not self.early_stops[i]]
                T = T[:int(self.n_configs / self.eta)]
                if backup_filename:
                    self.backup_data(backup_filename, s, i, T, chunk_offset)

                self.counter += 1


        return self.results


    def backup_data(self, backup_filename, s, i, T, chunk_offset):
        cur_state = {
            'results': self.results,
            'best_loss': self.best_loss,
            'best_counter': self.best_counter,
            'early_stops': self.early_stops,
            's': s,
            'i': i,
            'val_losses': self.val_losses,
            'counter': self.counter,
            'T': T,
            'chunk_offset': chunk_offset,
            'best_accuracy': self.best_accuracy,
            'best_params': self.best_params,
            'finished_threads': self.finished_threads
        }
        with open(backup_filename, 'wb') as f:
            pickle.dump(cur_state, f)
            print('state backuped')


    def gen_chunk(self, it, size):
        it = iter(it)
        return list(iter(lambda: tuple(islice(it, size)), ()))


    def try_params_in_thread(self, t, backup_filename, s, i, T, chunk_offset, chunks_list_len):
        start_time = time()

        result = self.model.try_params(self.n_iterations, t)

        seconds = int(round(time() - start_time))

        self.lock.acquire()
        try:
            self.finished_threads += 1
            self.print_progress(chunk_offset, chunks_list_len, seconds)
            loss = result['loss']
            self.val_losses.append(loss)

            early_stop = result.get('early_stop', False)
            self.early_stops.append(early_stop)

            acc = result['acc']
            if acc > self.best_accuracy:
                self.best_accuracy = acc

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_counter = self.counter
                self.best_params = t

            result['counter'] = self.counter
            result['params'] = t
            result['iterations'] = self.n_iterations
            self.results.append(result)

            if backup_filename:
                self.backup_data(backup_filename, s, i, T, chunk_offset)

        except Exception as e:
            print('ERROR', e)
        finally:
            self.lock.release()


    def print_progress(self, chunk_offset, chunks_list_len, seconds=-1):
        clear_output(wait=True)
        print('\n*** {} configurations x {:.1f} iterations each at all'.format(
                    self.n_configs, self.n_iterations ))
        print('\n{} | Threads finished {}/{} | Chunks finished: {}/{} | best accuracy: {} |lowest loss so far: {:.4f} (run {})\n'.format(
            self.counter, self.finished_threads, self.threads_num, chunk_offset, chunks_list_len, self.best_accuracy, self.best_loss, self.best_counter))
        print('Best params:')
        pprint(self.best_params)

        if seconds != -1:
            print(f'\nLast thread finished in {seconds} seconds')
