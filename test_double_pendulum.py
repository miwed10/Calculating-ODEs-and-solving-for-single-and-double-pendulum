from double_pendulum import (DoublePendulum, 
                             DoublePendulumResults)

import numpy as np
import pytest

def test_derivatives_at_rest_is_zero() -> None:
    """Tester at de deriverte til double pendulum er null når den er 
    i ro.
    Kjøres i terimanl: pytest -v test_double_pendulum.py
    """
    t = 0.0

    model = DoublePendulum()       
    u = np.array([0.0, 0.0, 0.0, 0.0])                 

    result = model(t, u)

    assert np.isclose(result[0], 0.0)
    assert np.isclose(result[1], 0.0)
    assert np.isclose(result[2], 0.0)
    assert np.isclose(result[3], 0.0)
    
@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0),
        (0, 0.5, 3.386187037),
        (0.5, 0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ],
)
def test_domega1_dt(theta1: float, theta2: float, expected: float) -> None:
    """Function for testing that domega_1/dt is computed correctly
    kjøres i terminal: pytest -v test_double_pendulum.py
    """
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = model(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)
    
@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5, -7.704787325),
        (0.5, 0, 6.768494455),
        (0.5, 0.5, 0.0),
    ],
)

def test_domega2_dt(theta1: float, theta2: float, expected: float) -> None:
    """Function for testing that domega_2/dt is computed correctly
    kjøres i terminal: pytest -v test_double_pendulum.py"""
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = model(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)
    
    
def test_solve_pendulum_ode_with_zero_ic() -> None:
    """Løser og tester at results fra solve i ODEModel for double pendulum ODE
    med null som inital condition gir null som resultat for alt.
    kjøres i terminal: pytest -v test_double_pendulum.py
    """
    
    model = DoublePendulum()    
    u0 = np.ndarray(shape = (4,), buffer = np.array([0.0, 0.0, 0.0, 0.0]))
    result = model.solve(u0 = u0, T = 2.0, dt = 0.01)
    
    assert np.allclose(result.solution, 0.0)

    
def test_solve_double_pendulum_function_zero_ic() -> None:
    """Tester at initial condition 0 gir 0 
    og at x1 = x2 = 0, y1 = -L1, y2 = -L1 - L2.
    kjøres i terminal: pytest -v test_double_pendulum.py
    """

    model = DoublePendulum()       
    u0 = np.ndarray(shape = (4,), buffer = np.array([0.0, 0.0, 0.0, 0.0]))               
    result = model.solve(u0 = u0, T = 2.0, dt = 0.01)

    ting = DoublePendulumResults(result.solution, time = 0.0, 
                                 L1 = model.L1, L2 = model.L2, g = model.g)
    
    assert np.allclose(result.solution, 0.0)

    assert np.all(ting.x1 == 0)
    assert np.all(ting.x2 == 0)
    assert np.all(ting.y1 == -ting.L1)
    assert np.all(ting.y2 == -ting.L1 - ting.L2)