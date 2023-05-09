import numpy as np
from typing import List
from .Boid import Boid
from .Rule import Rule


class Swarm:
    boids = []
    rules = []

    def __init__(self, grid: np.ndarray, nboids: int, rules: List[Rule]) -> None:
        self.boids = Boid(np.array([0, 0]), np.array([0, 0]), grid, 0)
        self.rules = rules

    def evolve() -> None:
        pass
