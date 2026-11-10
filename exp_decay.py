import numpy as np
import matplotlib.pyplot as plt
import ode

from ode import ODEModel
from scipy.integrate import solve_ivp

class ExponentialDecay(ODEModel):
    """
    Representerer en diffligning (ODE) på formen:
    du/dt = -au
    """

    @property
    def decay(self) -> float:
        """Returnerer decay-konstanten a"""
        return self._a
    
    @decay.setter
    def decay(self, value: float) -> None:
        """Endrer decay-konstanten a - valueError hvis ikke åoositiv verdi!"""
        if value < 0:
            raise ValueError(F"ExponentialDecay.__init__: a is negative ({value})")

        self._a = value

    def __init__(self, a: float):
        """a er decay-konstanten, kan ikke være negativ"""

        self.decay = a


    def __call__(self, t: float, u: np.ndarray[float]) -> np.ndarray[float]:   
        """Når vi kaller objektet som en funksjon, 
        regnes du/dt ut for disse veridene av t og u.
        (u kan være en array med mange elementer, 
        og du/dt vil ha like mange elementer
        """

        du_dt = -self.decay * u

        return du_dt
    
    @property
    def num_states(self) -> int:
        """Gir antall state variables i eksponential decay"""
        return 1
    
    @num_states.setter
    def num_states(self, value: float) -> None:
        """Gir AttributeError om man prøver å manuelt endre num_states"""
        raise AttributeError
    

if __name__ == "__main__":
    decay = ExponentialDecay(0.4)
    time_points = np.linspace(0, 10, 1001)

    results = solve_ivp(decay, (0, 10), [10], t_eval = time_points)
    solution = results.y[0]

    plt.plot(results.t, results.y[0], label = "ODE")
    plt.xlabel("tid")
    plt.ylabel("u(t)")
    plt.legend()
    plt.show()

    model = ExponentialDecay(0.4)
    result = model.solve(u0 = np.array([4.0]), T = 10.0, dt = 0.01)
    ode.plot_ode_solution(
        results=result, state_labels=["u"], filename="exponential_decay.png"
    )