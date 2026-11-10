import numpy as np
import matplotlib.pyplot as plt
import abc 

from scipy.integrate import solve_ivp
from typing import (NamedTuple,
                    Any,
                    Optional)

class InvalidInitialConditionError(RuntimeError):
    pass

class ODEResult(NamedTuple):
    """The result of solving an ODE
    
    Args: 
        time (np.ndarray): the time steps solved for
        solution (np.ndArray): the solution of the problme at the given times
    """

    time: np.ndarray
    solution: np.ndarray

    @property
    def num_states(self) -> int:
        """Returnerer en int som representere antall states"""
        return len(self.solution)

    @property
    def num_timepoints(self) -> int:
        """Returnerer en int som representerer antall timepoints"""
        return len(self.time)


class ODEModel(abc.ABC):
    """Felles grenesnitt for alle ODE-er (ordinære diffligninger)
    Kan ikke brukes direkte, må arves fra og det må implementeres en 
    løsningmetode for en bestemt type ODE
    """

    @abc.abstractmethod
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """Regne ut høyre side (RHS) av diffligningen du/dt = f(t, u).
        Må implementeres av klasser som arver herfra, ellers får du 
        NotImplementedError
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_states(self) -> int:
        """Gir nummer av state variables i ordinære diffligninger (ODE)"""
        raise NotImplementedError
    
    def _create_result(self, solutions: Any) -> ODEResult:
        """Systematiserer resultatet slik at det er lett å lese og kalle"""
        return ODEResult(time = solutions.t, solution = solutions.y)
    

    def solve(self, u0: np.ndarray, T: float, 
              dt: float, method: str = "RK45") -> ODEResult:
        """Funksjonen skal returnere løsningen til ODE problemet"""
        time_points = np.linspace(0, T, int(T/dt) + 1)
        solution = solve_ivp(self.__call__, (0, T), u0, t_eval = time_points, 
                            method = method
                            )

        return self._create_result(solution)
    
def plot_ode_solution(results: ODEResult,
                    state_labels: Optional[list[str]] = None,
                    filename: Optional[str] = None) -> None:
    """Funksjon som plotter ODE resultater, kan enten vise plottet eller 
    lagre det som en fil.
    """
    
    if state_labels == None:
        antall = len(results.solution)
        state_labels = []

        for i in range(1, antall + 1):  
            state_labels.append(f"State {i}")
    
    solution = results.solution

    for i, label in enumerate(state_labels):
        plt.plot(results.time, solution[i], label = label)
    plt.xlabel("time")
    plt.ylabel("ODE solutions")
    plt.legend()
    plt.grid()
    
    if filename != None:
        plt.savefig(filename)

    else:
        plt.show()