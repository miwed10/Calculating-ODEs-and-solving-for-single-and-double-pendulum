from exp_decay import ExponentialDecay
from scipy.integrate import solve_ivp
from ode import (ODEResult,
                 ODEModel,
                 InvalidInitialConditionError,
                 plot_ode_solution)

from pathlib import Path
from math import e

import numpy as np
import pytest
import ode

def test_rhs() -> None:
    """
    Tester høyre side (RHS) av diffligningen du/dt = -au
    Kjøres med: pytest -v test_exp_decay.py
    """
    t = 0.0

    model = ExponentialDecay(0.4)       
    u = np.array([3.2])                 

    du_dt = model(t, u)
    assert np.isclose(du_dt, -1.28)    

def test_negative_decay_raises_VauleError_constructor() -> None:
    """Tester at a ikke kan være negativ.
    Kjøres med: pytest -v test_exp_decay.py
    """
    a = -1.0

    with pytest.raises(ValueError):
        ExponentialDecay(a)


def test_negative_decay_raises_ValueError() -> None:
    """Tester at a ikke kan endres til å være negativ.
    kjøres ved: pytest -v test_exp_decay.py
    """
    a = 0.4
    model = ExponentialDecay(a)

    with pytest.raises(ValueError):
        model.decay = -1.0

def test_num_states() -> None:
    """Tester at antall state variables for ExpontialDecay er lik 1
    kjøres ved: pytest -v test_exp_decay.py
    """
    model = ExponentialDecay(a = 1.0)

    with pytest.raises(AttributeError):
         model.num_states = 2

    assert model.num_states == 1

def test_create_result_solution_needs_t_y() -> None:
    """Tester at solutions return tid og numerisk løsning, 
    kjøres ved: pytest -v test_exp_decay.py
    """
    decay = ExponentialDecay(0.4)
    time_points = np.linspace(0, 10, 1001)
    solution = solve_ivp(decay, (0, 10), [10.0], t_eval = time_points)

    del solution.t
    del solution.y

    with pytest.raises(AttributeError):
        solution.t

    with pytest.raises(AttributeError):
        solution.y

def test_solve_with_different_number_of_initial_states() -> None:
    """Tester at hvis u0 = np.ndarray har lengde 2 
    får vi InvalidInitialConditionError,
    kjøres ved: pytest -v test_exp_decay.py
    """
    u0 = np.array([1.3, 2.3])

    try: 
        len(u0) == 1

    except:
        raise InvalidInitialConditionError


@pytest.mark.parametrize(

    "a, u0,  T, dt",
    [
        (0.4, 2, 10, 0.01),
        (0.1, 3, 20, 0.001),
        (0.5, 4, 15, 1),                      
    ]

)

def test_solve_time(a, u0, T, dt) -> None:
    """Tester at tiden vi får resultater fra i solve er riktig,
        - første point er 0
        - siste point er T
        - forskjellen mellom andre og første point er dt,
    kjøres ved: pytest -v test_exp_decay.py
    """
    model = ExponentialDecay(a)
    solution = model.solve([u0], T, dt)

    assert solution.time[0] == 0
    assert solution.time[-1] == T

    differanse = solution.time[1] - solution.time[0]
    assert np.isclose(differanse, dt)


@pytest.mark.parametrize(

    "a, u0,  T, dt",
    [
        (0.4, 2, 10, 0.01),
        (0.1, 3, 20, 0.001),
        (0.5, 4, 15, 1),                      
    ]

)

def test_solve_solution(a, u0, T, dt) -> None:
    """Tester at output fra solve og det riktige svaret er riktig
    intill en margin på 1%.
    kjøres ved: pytest -v test_exp_decay.py
    """
    model = ExponentialDecay(a)
    solution = model.solve([u0], T, dt)

    midt_punkt = round(len(solution.time)/2)
    exact_verdi_mid = u0 * e**(-a * (midt_punkt * dt))
    computed_verdi_mid = solution.solution[0][midt_punkt] 

    relative_error_mid = (np.linalg.norm(computed_verdi_mid - exact_verdi_mid)
                           / np.linalg.norm(exact_verdi_mid))

    assert (relative_error_mid * 100) < 1

    exact_verdi_last = u0 * e**(-a * T)
    computed_verdi_last = solution.solution[0][-1] 

    relative_error_last = (np.linalg.norm(computed_verdi_last - exact_verdi_last)
                            / np.linalg.norm(exact_verdi_last))

    assert (relative_error_last * 100) < 1

def test_ODEResults() -> None:
    """Tester at ODEResults med en time array med lengde 3 og en solution array
    med shape (2, 3).
    kjøres ved: pytest -v test_exp_decay.py
    """
    results = ode.ODEResult(time=np.array([0, 1, 2]), solution=np.zeros((2, 3)))

    assert results.num_states == 2          
    assert results.num_timepoints == 3     


def test_plot_ode_solution_saves_file() -> None:
    """Tester om filen blir opprettet i plot_ode_solution.
    Kjøres ved: pytest -v test_exp_decay.py.
    """
    model = ExponentialDecay(0.4)
    result = model.solve(u0 = np.array([4.0]), T = 10.0, dt = 0.01)

    filename = Path("test_plot.png")
    if filename.is_file():
        filename.unlink()
    
    plot_ode_solution(result, filename = "test_plot.png")

    assert filename.is_file()
    filename.unlink()