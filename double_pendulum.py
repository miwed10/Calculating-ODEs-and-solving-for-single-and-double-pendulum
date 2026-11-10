from ode import ODEModel
from dataclasses import dataclass
from pendulum import plot_energy

import numpy as np
import math as m

@dataclass
class DoublePendulumResults:
    solution: np.ndarray
    time: np.ndarray
    L1: float
    L2: float
    g: float
    
    @property
    def theta1(self) -> np.ndarray:
        """Returnerer theta_1 over tid"""
        return [self.solution[i][0] for i in range(len(self.solution))]
    
    @property
    def omega1(self) -> np.ndarray:
        """Returnerer omega_1 over tid"""
        return [self.solution[i][1] for i in range(len(self.solution))]
    
    @property
    def theta2(self) -> np.ndarray:
        """Returnerer theta_2 over tid"""
        return [self.solution[i][2] for i in range(len(self.solution))]
    
    @property
    def omega2(self) -> np.ndarray:
        """Returnerer omega_2 over tid"""
        return [self.solution[i][3] for i in range(len(self.solution))]
    
    @property
    def x1(self) -> np.ndarray:
        """Returnerer x_1 over tid"""
        x1 = self.L1 * np.sin(self.theta1)
        return x1
    
    @property
    def y1(self) -> np.ndarray:
        """Returnerer y_1 over tid"""
        y1 = -self.L1 * np.cos(self.theta1)
        return y1
    
    @property
    def x2(self) -> np.ndarray:
        """Returnerer x_2 over tid"""
        x2 = self.x1 + self.L2 * np.sin(self.theta2)
        return x2
    
    @property
    def y2(self) -> np.ndarray:
        """Returnerer y_2 over tid"""
        y2 = self.y1 - self.L2 * np.cos(self.theta2)
        return y2
    
    @property
    def potential_energy(self) -> np.ndarray:
        """Returnerer array med potensiell energi over tid"""
        P1 = self.g * (self.y1 + self.L1)
        P2 = self.g * (self.y2 + self.L1 + self.L2)
        return P1 + P2
    
    @property
    def vx1(self) -> np.ndarray:
        """Returnerer en array av fart i x retning over tid"""
        vx1 = np.gradient(self.x1, self.time)
        return vx1
    
    @property
    def vy1(self) -> np.ndarray:
        """Returnerer en array av fart i y retning over tid"""
        vy1 = np.gradient(self.y1, self.time)
        return vy1
    
    @property
    def vx2(self) -> np.ndarray:
        """Returnerer en array av fart i x retning over tid"""
        vx2 = np.gradient(self.x2, self.time)
        return vx2
    
    @property
    def vy2(self) -> np.ndarray:
        """Returnerer en array av fart i y retning over tid"""
        vy2 = np.gradient(self.y2, self.time)
        return vy2
    
    @property
    def kinetic_energy(self) -> np.ndarray:
        """Returnerer array med kinetisk energi over tid"""
        K1 = 1/2 * (self.vx1**2 + self.vy1**2)
        K2 = 1/2 * (self.vx2**2 + self.vy2**2)
        return K1 + K2
    
    @property
    def total_energy(self) -> np.ndarray:
        """Returnerer array med total energi over tid"""
        E = self.potential_energy + self.kinetic_energy
        return E

class DoublePendulum(ODEModel):
    def __init__(self, L1: int = 1, L2: int = 1, g: float = 9.81):
        """Konstruktøren til DoublePendulum, tar inn L1,L2 og g som ahr 
        defalt verider L1 = L2 = 1, og g = 9.81.
        """
        self.L1 = L1
        self.L2 = L2
        self.g = g
        
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """Call funksjonen. 
        Tar inn u med variablene theta1, omega1, theta2, omega2.
        Returnerer de deriverte av de fire"""
        theta1, omega1,theta2, omega2 = u
        dtheta = theta2 - theta1
        
        dtheta1_dt = omega1
        domega1_dt = (self.L1* omega1**2 * np.sin(dtheta) * np.cos(dtheta) +
                      self.g * np.sin(theta2) * np.cos(dtheta) +
                      self.L2 * omega2**2 * np.sin(dtheta) -
                      2* self.g * np.sin(theta1)
                    ) / (
                    2 * self.L1 - self.L1 * np.cos(dtheta)**2
                    )
        
        dtheta2_dt = omega2
        domega2_dt = (-self.L2* omega2**2 * np.sin(dtheta) * np.cos(dtheta) + 
                      2 * self.g * np.sin(theta1) * np.cos(dtheta) - 
                      2* self.L1 * omega1**2 * np.sin(dtheta) -
                      2* self.g * np.sin(theta2)
                    ) / (
                    2 * self.L2 - self.L2 * np.cos(dtheta)**2
                    )
        
        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])
    
    def num_states(self) -> int:
        """Gir nummer av state variables i DoublePendulum"""
        return 4
    
    def _create_result(self, solutions) -> DoublePendulumResults:
        """Overkjører _create_result fra ODEModel"""
        return DoublePendulumResults(time = solutions.t, 
                                     solution = solutions.y.T,
                                     L1 = self.L1,
                                     L2 = self.L2,
                                     g = self.g)
        
if __name__ == "__main__":
    def exercise_3d() -> None:
        """Løser og plotter energien til dobbel pendulum over tid, 
        uten friksjon og luftmotstand.
        """
        model = DoublePendulum()
        u0 = np.ndarray(shape=(4,), buffer = np.array([m.pi/6, 0.35, 0, 0]))
        result = model.solve(u0 = u0, T = 10.0, dt = 0.01, method = "Radau")
        
        plot_energy(
            results = DoublePendulumResults(result.solution, result.time,
                                      L1 = model.L1, L2 = model.L2, g = model.g
                                      ), filename = "energy_double.png"
        )
    exercise_3d()