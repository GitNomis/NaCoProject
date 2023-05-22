from typing import Tuple, Union, Optional, overload
import numpy as np

from .state import State

class Environment:
    n_tiles:int
    grid:np.ndarray[State]
        
    def __init__(self,n_tiles:int,grid:np.ndarray[State]):
        self.n_tiles=n_tiles
        self.grid=grid   

    @classmethod
    def example(cls, size: Tuple[int, int], fire_size: Optional[Union[int, float]] = 1, water_size: Optional[Union[int, float]] = 1):
        """Create an environment containing a cellular automaton with a swarm of Boids superimposed on it. 

        Arguments:
            size (Tuple[int, int]): A tuple of size 2 containing the width and height of the grid (x, y).
            swarm_size (int): The number of Boids to simulate in the swarm.
            fire_size (Union[int, float]): The number of burning tiles that should be generated in the grid. 
            If ```fire_size``` < 1, then it represents a fraction of the tiles. Else, the fire size represents the number of burning tiles.
            water_size (Union[int, float]): The number of water tiles that should be generated in the grid. 
            If ```water_size``` < 1, then it represents a fraction of the tiles. Else, the water size represents the number of water tiles.

        Keyword Arguments:
            fire_size (Union[int, float]): The number of burning tiles that should be generated in the grid. (default: {1})
        """      
        n_tiles = size[0] * size[1]  
        grid = cls.create_grid(n_tiles=n_tiles,size=size, fire_size=fire_size, water_size=water_size)     
        return Environment(n_tiles,grid)

    @staticmethod
    def create_grid(n_tiles:int, size: Tuple[int, int], fire_size: Union[int, float], water_size: Union[int, float]) -> np.ndarray[State]:
        """Create a numpy array to represent a cellular automaton grid.

        Arguments:
            size (Tuple[int, int]): A tuple of size 2 containing the width and height of the grid (x, y).
            fire_size (Union[int, float]): The number of burning tiles that should be generated in the grid. 
            If ```fire_size``` < 1, then it represents a fraction of the tiles. Else, the fire size represents the number of burning tiles.
            water_size (Union[int, float]): The number of water tiles that should be generated in the grid. 
            If ```water_size``` < 1, then it represents a fraction of the tiles. Else, the water size represents the number of water tiles.

        Returns:
            np.ndarray: A representation of the cellular automaton. 
        """        
        grid = np.zeros(shape=size)
        n_burning_tiles = int(np.floor(n_tiles * fire_size)) if fire_size < 1 else fire_size
        n_water_tiles = int(np.floor(n_tiles * water_size)) if water_size < 1 else water_size

        xs = np.random.randint(0, size[1], size=n_burning_tiles+n_water_tiles)
        ys = np.random.randint(0, size[0], size=n_burning_tiles+n_water_tiles)

        fire_coordinates = (tuple(ys[:n_burning_tiles]), tuple(xs[:n_burning_tiles]))
        grid[fire_coordinates] = State.FIRE.value

        water_coordinates = (tuple(ys[n_burning_tiles:]), tuple(xs[n_burning_tiles:]))
        grid[water_coordinates] = State.WATER.value

        return grid        

    def update(self) -> None:
        """Update the environment.
        """        
        pass
    
    def calculate_fitness(self) -> int:
        """Calculate the fitness of the simulated environment, which is based on the number of burning tiles. 
        The maximal fitness is equal to the total number of tiles in the grid.

        Returns:
            int: the fitness of the simulated environment. 
        """        
        return self.n_tiles - np.sum(self.grid)
    
    def copy(self):
        return Environment(self.n_tiles,self.grid.copy())



