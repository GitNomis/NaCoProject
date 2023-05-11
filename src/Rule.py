import numpy as np
from typing import Optional
from sklearn.preprocessing import normalize


class Rule:
    weight = 1

    @classmethod
    def apply(self, swarm, velocities: np.ndarray) -> np.ndarray:
        pass

class Alignment(Rule):
    
    def __init__(self):
        super(Rule).__init__()

    def apply(self, swarm, velocities: np.ndarray) -> np.ndarray:
        force_vector = np.zeros(velocities.shape)
        for i, boid in enumerate(swarm.boids):
            neighbours_idx = swarm.kdtree.query_radius(boid.position[np.newaxis, :], r=swarm.vision_range)[0]
            force_vector[i] =  np.mean([b.velocity for b in swarm.boids[neighbours_idx]]) - boid.velocity
        return force_vector    
    
class Cohesion(Rule):
    strength = 1

    def __init__(self, strength: Optional[float] = 1):
        super(Rule).__init__()
        self.strength = strength

    def apply(self, swarm, velocities: np.ndarray) -> np.ndarray:
        force_vector = np.zeros(velocities.shape)
        for i, boid in enumerate(swarm.boids):
            neighbours_idx = swarm.kdtree.query_radius(boid.position[np.newaxis, :], r=swarm.vision_range)[0]
            force_vector[i] = np.mean([b.position for b in swarm.boids[neighbours_idx]]) - boid.position
        force_vector=normalize(force_vector, axis=1) * self.strength
        return force_vector
    
class Separation(Rule):
    strength = 1

    def __init__(self, strength: Optional[float] = 1):
        super(Rule).__init__()
        self.strength = strength

    def apply(self, swarm, velocities: np.ndarray) -> np.ndarray:
        force_vector = np.zeros(velocities.shape)
        for i, boid in enumerate(swarm.boids):
            dist, neighbours_idx = swarm.kdtree.query(boid.position[np.newaxis, :], k=2, sort_results=True)
            if neighbours_idx.shape[0] == 1 or dist[1] > swarm.vision_range:
                continue
            force_vector[i] = boid.position - swarm.boids[neighbours_idx[0, 1]].position
        force_vector=normalize(force_vector, axis=1) * self.strength
        return force_vector    