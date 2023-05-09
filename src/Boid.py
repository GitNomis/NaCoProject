import numpy as np

from .State import State

class Boid:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, grid: np.ndarray, size: float) -> None:
        self.position = position
        self.velocity = velocity
        self.size = size
        self.grid = grid

        self.carrying_water = False

    def update(self) -> None:
        if not self.carrying_water:
            self.get_water()

        # Swarm updates velocity vector, so boid does not think.
        self.position += self.velocity

        self.wrap()

    def get_water(self) -> None:
        if(self.grid[self.center_x, self.center_y] == State.WATER.value):
            self.carrying_water = True

    def wrap(self) -> None:
        width = self.grid.shape[1]
        height = self.grid.shape[0]

        if self.position[0] > width:
            self.position[0] -= width
        if self.position[0] < 0:
            self.position[0] += width

        if self.position[1] > height:
            self.position[1] -= height
        if self.position[1] < 0:
            self.position[1] += height