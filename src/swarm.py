from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .rule import Rule

from typing import List
import numpy as np
from sklearn.neighbors import KDTree
import random

from .boid import Boid


class Swarm:
    boids:List[Boid]
    rules:List[Rule]
    vision_range:int
    kdtree:KDTree

    def __init__(self, grid: np.ndarray[int], vision_range: float, nboids: int, rules: List[Rule]) -> None:
        """ Swarm class representing a group of boids with shared behaviour

        Args:
            grid (np.ndarray[int]): the grid the swarm lives on
            vision_range (float): the vision range of the boids
            nboids (int): the number of boids within the swarm
            rules (List[Rule]): the list of rules that the boids follow
        """
        self.boids = np.array([Boid(np.array(np.random.uniform([0, grid.shape[0]-1.01])), np.array([1, 1], dtype=float), grid, 1) for _ in range(nboids)])
        self.rules = rules
        self.vision_range = vision_range
        self.kdtree = self.construct_KDTree()

    def construct_KDTree(self) -> KDTree:
        return KDTree([boid.position for boid in self.boids])
    
    def update(self) -> None:
        velocities = np.array([boid.velocity for boid in self.boids])
        force_vector = np.zeros(velocities.shape)

        for rule in self.rules:
            force_vector += rule.weight * rule.apply(self, velocities)

        for i, boid in enumerate(self.boids):
            boid.velocity += force_vector[i]
            boid.update()
            
        self.kdtree=self.construct_KDTree()
    
    def evolve() -> None:
        pass

