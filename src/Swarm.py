from typing import List

import numpy as np
from sklearn.neighbors import KDTree
from matplotlib.markers import MarkerStyle
import random

from .Boid import Boid
from .Rule import Rule


class Swarm:
    boids = []
    rules = []
    vision_range = 0
    kdtree = None

    def __init__(self, grid: np.ndarray, vision_range: float, nboids: int, rules: List[Rule]) -> None:
        self.boids = np.array([Boid(np.array(np.random.uniform([0, grid.shape[0]-1.01])), np.array([1, 1], dtype=float), grid, 1) for _ in range(nboids)])
        self.rules = rules
        self.vision_range = vision_range
        self.kdtree = self.construct_KDTree()

    def construct_KDTree(self) -> KDTree:
        return KDTree([boid.position for boid in self.boids])
    
    def update(self) -> None:
        velocities = np.array([boid.velocity for boid in self.boids])
        force_vector = np.zeros(velocities.shape)

        # FIXME: Boids cannot "look around the corner", behaviour changes when boids wrap to other side of the field
        for rule in self.rules:
            force_vector += rule.weight * rule.apply(self, velocities)

        for i, boid in enumerate(self.boids):
            boid.velocity += force_vector[i]
            boid.update()
            
        self.kdtree=self.construct_KDTree()
        
        # Animation -------------------
        offsets = []
        markers = []
        
        for boid in self.boids:
            offsets.append([boid.position[0], boid.position[1]])
            marker = MarkerStyle(">")
            marker._transform = marker.get_transform().rotate_deg(np.angle(complex(*boid.direction),True))
            marker._transform = marker.get_transform().scale(2, 2)
            markers.append(marker)
        
        # -----------------------------
        return offsets, markers
    
    def evolve() -> None:
        pass

