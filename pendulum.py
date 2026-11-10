from ode import ODEModel
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import math as m
import ode

@dataclass
class PendulumResults:
    """Results from solving the pendulum problem.
    
    Args:
        time (np.ndarray): The timesteps of the solution.
        solution (np.ndarray): The values of the solution at the given timesteps.
        L (float): The length of the pendulum rod.
        g (float): The gravitational acceleration.
    """
    solution: np.ndarray
    time: np.ndarray
    L: float
    g: float
    
    @property
    def theta(self) -> np.ndarray:
        """Returnerer pendulumvinkelen over tid"""
        return [self.solution[i][0] for i in range(len(self.solution))]
    
    @property
    def omega(self) -> np.ndarray:
        """Returerer vinkelfarten over tid"""
        return [self.solution[i][1] for i in range(len(self.solution))]
    
    @property
    def x(self) -> np.ndarray:
        """Returnerer x-coordinatet over tid"""
        x = self.L*np.sin(self.theta)
        return x
    
    @property 
    def y(self) -> np.ndarray:
        """Returnerer y-coordinatet over tid"""
        y = -self.L*np.cos(self.theta)
        return y
    
    @property
    def potential_energy(self) -> np.ndarray:
        """Returnerer en array av potensiell energi til systemetz over tid"""
        P = self.g * (self.y + self.L)
        return P
    
    @property
    def vx(self) -> np.ndarray:
        """Returnerer en array av fart i x retning over tid"""
        vx = np.gradient(self.x, self.time)
        return vx
    
    @property
    def vy(self) -> np.ndarray:
        """Returnerer en array av fart i y retning over tid"""
        vy = np.gradient(self.y, self.time)
        return vy
    
    @property
    def kinetic_energy(self) -> np.ndarray:
        """Returnerer en array med kinetisk energi til systemet over tid"""
        K = 1/2*(self.vx**2 + self.vy**2)
        return K
    
    @property
    def total_energy(self) -> np.ndarray:
        """Returnerere en array med total energien til systemet over tid"""
        E = self.kinetic_energy + self.potential_energy
        return E
        
    
class Pendulum(ODEModel):
    """Representerer det fysiske sytemet dannet av en pendel
    """

    def __init__(self, L: int = 1, g: float = 9.81):
        """The constructor, tar inn L og g, har defalt verider L = 1, 
        og g = 9.81.
        """

        self.g = g
        self.L = L

        
    def __call__(self, t: float, u: np.ndarray[float, float]) -> np.ndarray[float]:
        """Call funksjon, la the order of states in the 
        input array u be (θ, ω).
        """
        self.theta = u[0]
        self.omega = u[1]
        dθ_dt = self.omega
        dω_dt = -(self.g/self.L) * m.sin(self.theta)

        return dθ_dt, dω_dt


    def num_states(self) -> int:
        """Gir nummer av state variables i ordinære diffligninger (ODE)"""
        return 2
    
    def _create_result(self, solutions) -> PendulumResults:
        """Overkjører ODEResult for å returnere PendulumResults"""
        return PendulumResults(time = solutions.t, 
                               solution = solutions.y.T, 
                               L = self.L,
                               g = self.g)


class DampenedPendulum(Pendulum):
    def __init__(self, B: float, L: int = 1, g: float = 9.81):
        """The constructor av DampenedPendulum, definerer B, L og g, 
        hvor L og g har defalt verider (1, 9.81)
        """
        self.g = g
        self.L = L
        self.B = B
    
    def __call__(self, t: float, u: np.ndarray[float, float]) -> np.ndarray[float]:
        """Call funksjon, la the order of states in the 
        input array u be (θ, ω). Tar hensyn til luftmotstand.
        """
        self.theta = u[0]
        self.omega = u[1]
        dθ_dt = self.omega
        dω_dt = -(self.g/self.L) * m.sin(self.theta) - self.B*self.omega

        return dθ_dt, dω_dt
        

def plot_energy(results: PendulumResults, filename: Optional[str] = None) -> None:
        """Plotter kinetisk, potensiell og total energi"""
        plt.plot(results.time, results.kinetic_energy, label = "kinetic energy")
        plt.plot(results.time, results.potential_energy, label = "potential energy")
        plt.plot(results.time, results.total_energy, label = "total energy")
        
        plt.xlabel("time")
        plt.ylabel("Energy functions")
        plt.legend()
        plt.grid()
    
        if filename != None:
            plt.savefig(filename)

        else:
            plt.show()
        
        
if __name__ == "__main__":
    def exercise_2b() -> None:
        """Løser pendulum og løser ODE-ene med å bruke solve fra ODEModel. 
        Skal plotte resultatet og lagre det til en fil, exercise_2b.png
        """

        model = Pendulum()
        u0 = np.ndarray(shape=(2,), buffer = np.array([m.pi/6, 0.35]))
        result = model.solve(u0 = u0, T = 10.0, dt = 0.01)
        ode.plot_ode_solution(
            results=result, state_labels=["θ", "ω"], filename="exercise_2b.png"
        )
    
    def exercise_2g() -> None:
        """Løser og plotter energien til pendulumet over tid, dette er 
        med utelokking av friksjon og luftmotstand
        """
        model = Pendulum()
        u0 = np.ndarray(shape=(2,), buffer = np.array([m.pi/6, 0.35]))
        result = model.solve(u0 = u0, T = 10.0, dt = 0.01)
        
        plot_energy(
            results = PendulumResults(result.solution, result.time,
                                      L = model.L, g = model.g
                                      ), filename = "energy_single.png"
        )
        
    def exercise_2h() -> None:
        """Løser og plotter energien til pendulumet over tid, men denne
        gangen med friksjon og luftmotstand.
        """
        model = DampenedPendulum(B = 1)
        u0 = np.ndarray(shape=(2,), buffer = np.array([m.pi/6, 0.35]))
        result = model.solve(u0 = u0, T = 10.0, dt = 0.01)
        
        plot_energy(
            results = PendulumResults(result.solution, result.time,
                                      L = model.L, g = model.g
                                      ), filename = "energy_dampened.png"
        )
        
    exercise_2h()