from pendulum import Pendulum, PendulumResults
from math import pi

import numpy as np

def test_pendulum_class() -> None:
    """Tester beregneringer i Pendulum klassen
    Kjøres ved: pytest -v test_pendulum.py
    """
    t = 0.0

    model = Pendulum(L = 1.42)       
    u = np.array([pi/6, 0.35])                 

    result = model(t, u)

    assert np.isclose(result[0], 0.35)
    assert np.isclose(result[1], -3.45422)

def test_pendulum_class_at_rest() -> None:
    """Tester at Pendulum klasse returnerer 0 for begge variabler når
     start verdiene er null
     Kjøres ved: pytest -v test_pendulum.py
    """
    t = 0.0

    model = Pendulum(L = 1.42)       
    u = np.array([0.0, 0.0])                 

    result = model(t, u)

    assert np.isclose(result[0], 0.0)
    assert np.isclose(result[1], 0.0)


def test_solve_pendulum_ode_with_zero_ic() -> None:
    """Tester at solution fra solve for Pendulum returnerer bare 0 for
    vinkel og vinkelfart.
    Kjøres ved: pytest -v test_pendulum.py
    """
    model = Pendulum()
    u0 = np.ndarray(shape=(2,), buffer = np.array([0.0, 0.0]))
    result = model.solve(u0 = u0, T = 2.0, dt = 0.01)

    assert np.allclose(result.solution, 0.0)
        

def test_solve_pendulum_functiuon_zero_ic() -> None:
    """Tester at Pendulum returnerer 0 for vinkel, vinkelfart, x-coordinat,
    og -L for y-coordinatet.
    Kjøres ved: pytest -v test_pendulum.py
    """
    model = Pendulum(L = 1)
    u0 = np.ndarray(shape=(2,), buffer = np.array([0.0, 0.0]))
    result = model.solve(u0 = u0, T = 2.0, dt = 0.01)

    ting = PendulumResults(result.solution, result.time, L = 1, g = 9.81)
    
    assert np.allclose(result.solution, 0.0)

    assert np.all(ting.x == 0.0)
    assert np.all(ting.y == -ting.L)