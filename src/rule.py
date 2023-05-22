from __future__ import annotations
from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from .swarm import Swarm

import numpy as np
from typing import Optional
from sklearn.preprocessing import normalize


class Rule:
    """
    Rule interface:
        Every Rule has a weight and an apply method that enforces a rule on a given swarm

    """
    weight: float
    std: float

    def __init__(self, weight: float = 1.0, std: float = 0.1):
        self.weight = weight
        self.std = std

    def apply(self, swarm: Swarm, velocities: np.ndarray[float]) -> np.ndarray[float]:
        pass

    @staticmethod
    def crossover(rule: Rule, other: Rule) -> Rule:
        mean = rule.weight if np.random.random() > 0.5 else other.weight
        std = rule.std if np.random.random() > 0.5 else other.std
        weight = np.random.normal(mean, std)
        return Rule(weight=weight)

    def mutate(self) -> None:
        self.weight = np.random.normal(self.weight, 2*self.std)

    def __str__(self):
        return f"weight = {self.weight}"    


class Alignment(Rule):

    def __init__(self, **params):
        super().__init__(**params)

    def apply(self, swarm: Swarm, velocities: np.ndarray[float]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)
        for i, boid in enumerate(swarm.boids):
            neighbours_idx = swarm.kdtree.query_radius(
                boid.position[np.newaxis, :], r=swarm.vision_range)[0]
            force_vector[i] = np.mean(
                [b.velocity for b in swarm.boids[neighbours_idx]]) - boid.velocity
        return force_vector

    @staticmethod
    def crossover(rule: Alignment, other: Alignment) -> Alignment:
        new_rule = super().crossover(rule, other)
        new_rule.__class__ = Alignment

        return new_rule
    
    def __str__(self):
        return f"Alignment: {super().__str__()}"

    


class Cohesion(Rule):
    strength: float
    std_strength: float = 0.1

    def __init__(self, strength: Optional[float] = 1, **params):
        super().__init__(**params)
        self.strength = strength

    def apply(self, swarm: Swarm, velocities: np.ndarray[float]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)
        for i, boid in enumerate(swarm.boids):
            neighbours_idx = swarm.kdtree.query_radius(
                boid.position[np.newaxis, :], r=swarm.vision_range)[0]
            force_vector[i] = np.mean(
                [b.position for b in swarm.boids[neighbours_idx]]) - boid.position
        force_vector = normalize(force_vector, axis=1) * self.strength
        return force_vector

    @staticmethod
    def crossover(rule: Cohesion, other: Cohesion) -> Cohesion:
        new_rule = super().crossover(rule,other)
        new_rule.__class__ = Cohesion

        mean = rule.strength if np.random.random() > 0.5 else other.strength
        std = rule.std_strength if np.random.random() > 0.5 else other.std_strength
        new_rule.strength = np.random.normal(mean, std)

        return new_rule
    
    def __str__(self):
        return f"Cohesion: {super().__str__()}, strength = {self.strength}"


class Separation(Rule):
    strength: float
    std_strength: float = 0.1

    def __init__(self, strength: Optional[float] = 1, **params):
        super().__init__(**params)
        self.strength = strength

    def apply(self, swarm: Swarm, velocities: np.ndarray[float]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)
        for i, boid in enumerate(swarm.boids):
            dist, neighbours_idx = swarm.kdtree.query(
                boid.position[np.newaxis, :], k=2, sort_results=True)
            if neighbours_idx.shape[1] == 1 or dist[0, 1] > swarm.vision_range:
                continue
            force_vector[i] = boid.position - swarm.boids[neighbours_idx[0, 1]].position
        force_vector = normalize(force_vector, axis=1) * self.strength
        return force_vector

    @staticmethod
    def crossover(rule: Separation, other: Separation) -> Separation:
        new_rule = super().crossover(rule,other)
        new_rule.__class__ = Separation

        mean = rule.strength if np.random.random() > 0.5 else other.strength
        std = rule.std_strength if np.random.random() > 0.5 else other.std_strength
        new_rule.strength = np.random.normal(mean, std)

        return new_rule

    def __str__(self):
        return f"Separation: {super().__str__()}, strength = {self.strength}"
