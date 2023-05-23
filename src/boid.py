from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .evolution import Environment
from .state import State

class Boid:
    position:np.ndarray[float]
    velocity:np.ndarray[float]
    size:float
    env:Environment
    carrying_water:bool

    def __init__(self, position: np.ndarray[float], velocity: np.ndarray[float], env:Environment, size: float) -> None:
        """Boid class representing a single cute boid

        Args:
            position (np.ndarray): the boids starting position on the grid
            velocity (np.ndarray): the boids starting velocity
            env (Environment): the environment the swarm lives in 
            size (float): the size of the boid
        """
        self.position = position
        self.velocity = velocity
        self.size = size
        self.env = env
        self.pickup_chance = 0.5

        self.carrying_water = False

    def update(self) -> None:
        """Update the behaviour of the Boid given the environment.
        """        
        # Swarm updates velocity vector, so boid does not think.
        self.position += self.velocity * 0.1

        if not self.border_handling():

            # Extinguish fire
            if self.on_fire() and self.carrying_water:
                self.extinguish_fire()
            
            if not self.carrying_water and np.random.random() < self.pickup_chance:
                self.get_water()
        else:
            self.position += self.velocity * 0.2

    def on_fire(self) -> bool:
        """Verifies whether the Boid is located on a fire tile.

        Returns:
            bool: True if the Boid is located on a fire tile.
        """        
        return self.env.grid[int(self.position[0]), int(self.position[1])] == State.FIRE.value

    def extinguish_fire(self) -> None:
        """Extinguishes fire by dropping water, changing the grid by reference and setting the attribute ```carrying_water``` to false.
        """        
        self.env.grid[int(self.position[0]), int(self.position[1])] = State.BARREN.value
        self.env.n_fires-=1
        self.carrying_water = False

    def get_water(self) -> None:
        """Collect water if the boid is hovering above a water tile.
        """        
        if(self.env.grid[int(self.position[0]), int(self.position[1])] == State.WATER.value):
            self.carrying_water = True

    def border_handling(self) -> bool:
        '''
        Bounces the boid back off the borner
        '''
        if self.position[0] > self.env.grid.shape[0] or self.position[0] < 0:
            self.velocity *= np.array([-1, 1])
            return True

        if self.position[1] > self.env.grid.shape[1] or self.position[1] < 0:
            self.velocity *= np.array([1, -1])
            return True
        
        return False
    
    def rigid_border(self, position: np.ndarray[float]) -> np.ndarray[float]:
        """Prevent the boid from going out of bounds. 

        Arguments:
            position (np.array[float]) -- The boid's x, y position

        Returns:
            the corrected position of the boid
        """        
        for i in range(position.shape[0]):
            position[i] = max(position[i], 0)
            position[i] = min(position[i], self.env.grid.shape[0])
        return position