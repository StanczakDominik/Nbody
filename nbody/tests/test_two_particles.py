import pytest
import numpy as np
from nbody.run_nbody import Simulation

def dt(request):
    return request.param

@pytest.fixture(params=[1e-7, 1e-6, 1e-5])
def simulation(request):
    sim = Simulation({}, 2, 100, request.param, file_path=None, m=1, T=0, dx=1, save_every_x_iters=1, gpu=False)
    sim.r = np.array([[0,0,0], [1,0,0]], dtype=float)
    sim.extrapolate_old_r()
    sim.run()
    return sim

@pytest.fixture(params=[1e-4, 1e-3, 1e-2])
def unstable_simulation(request):
    sim = Simulation({}, 2, 100, request.param, file_path=None, m=1, T=0, dx=1, save_every_x_iters=1, gpu=False)
    sim.r = np.array([[0,0,0], [1,0,0]], dtype=float)
    sim.extrapolate_old_r()
    sim.run()
    return sim

def plot(df):
    import matplotlib.pyplot as plt
    labels = [x for x in df.columns if 'energy' in x]
    fig, ax = plt.subplots()
    for label in labels:
        coefficients = np.polyfit(np.log(df['t']), np.log(df[label]), deg=3)
        poly = np.poly1d(coefficients)
        yfit = lambda x: np.exp(poly(np.log(df['t'])))
        df[f"{label} fit: {coefficients[1]:.3f}"] = yfit(df['t'])
    labels = [x for x in df.columns if 'energy' in x]
    df.plot(x='t', y=labels, logy=True, logx=True, ax=ax)
    plt.legend()
    plt.show()

def test_exponent_scaling(simulation):
    df = np.abs(simulation.diagnostic_df()).iloc[1:]
    label = "total_energy"
    coefficients = np.polyfit(np.log(df['t']), np.log(df[label]), deg=3)
    condition = 0 < coefficients[1] < 0.1
    if not condition:
        plot(df)
    assert condition
    
@pytest.mark.xfail
def test_unstable_exponent_scaling(unstable_simulation):
    df = np.abs(unstable_simulation.diagnostic_df()).iloc[1:]
    label = "total_energy"
    coefficients = np.polyfit(np.log(df['t']), np.log(df[label]), deg=3)
    condition = 0 < coefficients[1] < 0.1
    if condition:
        plot(df)
    assert condition

