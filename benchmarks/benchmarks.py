from nbody.run_nbody import Simulation

class SequentialIterationSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    timeout=1000
    params = ['python', 'njit', 'njit_parallel']
    def setup(self, engine):
        np.random.seed(0)
        self.sim = Simulation({}, n, 100, 1e-6, file_path=None, m=1, T=0, dx=1, save_every_x_iters=1, gpu=False, shape='fcc', engine=engine)
    
    def time_run(self):
        self.sim.run()
