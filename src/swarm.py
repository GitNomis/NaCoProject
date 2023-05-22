from __future__ import annotations
from typing import TYPE_CHECKING

from sklearn.preprocessing import normalize
if TYPE_CHECKING:
    from .rule import Rule
    from .evolution import Environment


from typing import List,Dict
import numpy as np
from sklearn.neighbors import KDTree
import random

from .boid import Boid


class Swarm:
    boids:List[Boid]
    rules:Dict[object,Rule]
    vision_range:int
    kdtree:KDTree
    env:Environment

    def __init__(self, env: Environment, vision_range: float, nboids: int, rules: List[Rule]) -> None:
        """ Swarm class representing a group of boids with shared behaviour

        Args:
            grid (np.ndarray[int]): the grid the swarm lives on
            vision_range (float): the vision range of the boids
            nboids (int): the number of boids within the swarm
            rules (List[Rule]): the list of rules that the boids follow
        """
        self.boids = np.array([Boid(np.array(np.random.uniform(0, env.grid.shape[0]-1.01, 2)), np.random.uniform(-1, 1, 2), env.grid, 1) for _ in range(nboids)])
        self.rules = dict((type(r), r) for r in rules)
        self.vision_range = vision_range
        self.kdtree = self.construct_KDTree()
        self.env = env

    def construct_KDTree(self) -> KDTree:
        return KDTree([boid.position for boid in self.boids])
    
    def update(self) -> None:
        velocities = np.array([boid.velocity for boid in self.boids])
        neighbours_idx,_=self.kdtree.query_radius([boid.position for boid in self.boids],r=self.vision_range,return_distance=True,sort_results=True)
        force_vector = np.zeros(velocities.shape)

        for rule in self.rules.values():
            force_vector += rule.weight * rule.apply(self, velocities,neighbours_idx)

        for i, boid in enumerate(self.boids):
            boid.velocity += force_vector[i]
            if np.linalg.norm(boid.velocity) > 4:
               boid.velocity=normalize([boid.velocity],axis=1)[0]*4
            boid.update()

        self.env.update()    
        
