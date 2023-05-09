from enum import Enum
from typing import Tuple, Union, Optional, List
import numpy as np

from .Swarm import Swarm
from .State import State



class Environment:
    def __init__(self, size: Tuple[int, int], swarm_size: int, fire_size: Optional[Union[int, float]] = 1):
        """Create an environment containing a cellular automaton with a swarm of Boids superimposed on it. 

        Arguments:
            size (Tuple[int, int]): A tuple of size 2 containing the width and height of the grid (x, y).
            swarm_size (int): The number of Boids to simulate in the swarm.

        Keyword Arguments:
            fire_size (Union[int, float]): The number of burning tiles that should be generated in the grid. (default: {1})
        """      
        self.n_tiles = size[0] * size[1]  
        self.grid = self.create_grid(size=size, fire_size=fire_size)
        rules = None  # TODO: initialize rules
        self.swarm = Swarm(grid=self.grid, nboids=size, rules=rules)


    def create_grid(self, size: Tuple[int, int], fire_size: Union[int, float]) -> np.ndarray:
        """Create a numpy array to represent a cellular automaton grid.

        Arguments:
            size (Tuple[int, int]): A tuple of size 2 containing the width and height of the grid (x, y).
            fire_size (Union[int, float]): The number of burning tiles that should be generated in the grid. 
            If ```fire_size``` < 1, then it represents a percentage of the tiles. Else, the fire size represents the number of burning tiles.

        Returns:
            np.ndarray: A representation of the cellular automaton. 
        """        
        grid = np.zeros(shape=size)
        n_burning_tiles = int(np.floor(self.n_tiles * fire_size)) if fire_size < 1 else fire_size

        xs = np.random.randint(low=0, high=size[1], size=n_burning_tiles)
        ys = np.random.randint(low=0, high=size[0], size=n_burning_tiles)

        fire_coordinates = (tuple(ys), tuple(xs))
        grid[fire_coordinates] = State.FIRE.value
        return grid
    
    def calculate_fitness(self) -> int:
        """Calculate the fitness of the simulated environment, which is based on the number of burning tiles. 
        The maximal fitness is equal to the total number of tiles in the grid.

        Returns:
            int: the fitness of the simulated environment. 
        """        
        return self.n_tiles - np.sum(self.grid)


