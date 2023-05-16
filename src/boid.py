import numpy as np

from .state import State

class Boid:
    position:np.ndarray[float]
    velocity:np.ndarray[float]
    size:float
    grid:np.ndarray[State]
    carrying_water:bool

    def __init__(self, position: np.ndarray[float], velocity: np.ndarray[float], grid: np.ndarray[State], size: float) -> None:
        """Boid class representing a single cute boid

        Args:
            position (np.ndarray): the boids starting position on the grid
            velocity (np.ndarray): the boids starting velocity
            grid (np.ndarray): the environments grid 
            size (float): the size of the boid
        """
        self.position = position
        self.velocity = velocity
        self.size = size
        self.grid = grid
        self.pickup_chance = 0.5

        self.carrying_water = False

    def update(self) -> None:
        """Update the behaviour of the Boid given the environment.
        """        
        # Swarm updates velocity vector, so boid does not think.
        self.position += self.velocity * 0.1

        print(self.carrying_water)

        if not self.border_handling():

            # Extinguish fire
            if self.on_fire() and self.carrying_water:
                self.extinguish_fire()
            
            # FIXME: Solve rounding causing out-of-bounds errors
            if not self.carrying_water and np.random.random() < self.pickup_chance:
                self.get_water()

    def on_fire(self) -> bool:
        """Verifies whether the Boid is located on a fire tile.

        Returns:
            bool: True if the Boid is located on a fire tile.
        """        
        return self.grid[int(self.position[0]), int(self.position[1])] == State.FIRE.value

    def extinguish_fire(self) -> None:
        """Extinguishes fire by dropping water, changing the grid by reference and setting the attribute ```carrying_water``` to false.
        """        
        self.grid[int(self.position[0]), int(self.position[1])] = State.BARREN.value
        self.carrying_water = False

    def get_water(self) -> None:
        """Collect water if the boid is hovering above a water tile.
        """        
        if(self.grid[int(self.position[0]), int(self.position[1])] == State.WATER.value):
            self.carrying_water = True

    def border_handling(self) -> bool:
        '''
        Bounces the boid back off the borner
        '''
        if self.position[0] > self.grid.shape[0] or self.position[0] < 0:
            self.velocity *= np.array([-1, 1])
            return True

        if self.position[1] > self.grid.shape[1] or self.position[1] < 0:
            self.velocity *= np.array([1, -1])
            return True
        
        return False