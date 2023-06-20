from __future__ import annotations
from typing import TYPE_CHECKING

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
    vision_range:float
    max_speed:float
    kdtree:KDTree
    env:Environment

    def __init__(self, env: Environment, vision_range: float, max_speed:float, nboids: int, rules: List[Rule]) -> None:
        """ Swarm class representing a group of boids with shared behaviour

        Args:
            env (Environment): the environment the swarm lives in
            vision_range (float): the vision range of the boids
            nboids (int): the number of boids within the swarm
            rules (List[Rule]): the list of rules that the boids follow
        """
        self.boids = np.array([Boid(np.random.uniform(0, env.grid.shape[0]-1.01, 2), np.random.uniform(-1, 1, 2)*max_speed, env, 1) for _ in range(nboids)])
        self.rules = dict((type(r), r) for r in rules)
        self.vision_range = vision_range
        self.max_speed = max_speed
        self.kdtree = self.construct_KDTree()
        self.env = env

    def reset(self,env:Environment) -> None:
        """Reset the swarm and set a new environment. 
        Arguments:
            env (Environment): An environment.

        """   
        self.boids = np.array([Boid(np.random.uniform(0, env.grid.shape[0]-1.01, 2), np.random.uniform(-1, 1, 2)*self.max_speed, env, 1) for _ in self.boids])
        self.kdtree = self.construct_KDTree()
        self.env=env


    def construct_KDTree(self) -> KDTree:
        """Construct a KDTree for the boid positions.

        Returns:
            KDTree: a KDTree for the boid positions.  
        """   
        return KDTree([boid.position for boid in self.boids])
    
    def simulate(self, n_iters: int) -> int:
        for iter in range(n_iters):
            self.update()
            if not self.env.contains_fire():
                return (n_iters - iter) + self.env.calculate_fitness()
        return self.env.calculate_fitness()
    
    def update(self) -> None:
        """Update the boids in the swarm. 
        """   
        velocities = np.array([boid.velocity for boid in self.boids])
        positions = np.array([boid.position for boid in self.boids])
        force_vector = np.zeros(velocities.shape)
        
        neighbours_idx,_=self.kdtree.query_radius(positions,r=self.vision_range,return_distance=True,sort_results=True)
        for i, n_idx in enumerate(neighbours_idx):
            neighbours_idx[i] = n_idx[n_idx!=i]        

        for rule in self.rules.values():
            force_vector += rule.weight * rule.apply(self,positions, velocities,neighbours_idx)

        for i, boid in enumerate(self.boids):
            boid.velocity += force_vector[i]
            if np.linalg.norm(boid.velocity) > self.max_speed:
               boid.velocity *= self.max_speed / np.linalg.norm(boid.velocity)
            boid.update()
            
        self.kdtree=self.construct_KDTree()
        self.env.update()    
        
