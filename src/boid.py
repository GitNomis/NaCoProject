import numpy as np

from .state import State

class Boid:
    position:np.ndarray[float]
    velocity:np.ndarray[float]
    size:float
    grid:np.ndarray[State]
    direction:np.ndarray[float]
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

        self.direction = np.random.uniform(low=-1, high=1, size=2)
        self.carrying_water = False

    def update(self) -> None:
        # Swarm updates velocity vector, so boid does not think.
        self.position += self.velocity * 0.1
        self.wrap()
        
        # TODO: Add updating direction for plotting
        
        # FIXME: Solve rounding causing out-of-bounds errors
        # if not self.carrying_water:
        #     self.get_water()

    def get_water(self) -> None:
        if(self.grid[int(round(self.position[0])), int(round(self.position[1]))] == State.WATER.value):
            self.carrying_water = True

    def wrap(self) -> None:
        width = self.grid.shape[0]
        height = self.grid.shape[1]

        if self.position[0] > width:
            self.position[0] -= width
        if self.position[0] < 0:
            self.position[0] += width

        if self.position[1] > height:
            self.position[1] -= height
        if self.position[1] < 0:
            self.position[1] += height