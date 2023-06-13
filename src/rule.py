from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .swarm import Swarm
    from .boid import Boid

import numpy as np
from typing import Optional
from sklearn.neighbors import KDTree

from .state import State


class Rule:
    """
    Rule interface:
        Every Rule has a weight and an apply method that enforces a rule on a given swarm

    """
    weight: float
    std: float

    def __init__(self, weight: float = 1.0, std: float = 0.2):
        self.weight = weight
        self.std = std

    def apply(self, swarm: Swarm, positions: np.ndarray[float], velocities: np.ndarray[float],neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        pass

<<<<<<< HEAD

    # TODO: I put this in rule now, but perhaps that's not the best/cleanest location
    def to_closest_tile(self, vision_range: float, tree: KDTree, boid: Boid) -> np.ndarray[float]:
        fire_within_vision_idx, _ = tree.query_radius([boid.position], r=vision_range, sort_results=True, return_distance=True)
        if len(fire_within_vision_idx[0]) > 0:
            closest_fire = tree.data[fire_within_vision_idx[0][0]]
            force = closest_fire - boid.position
        else:
            force = np.zeros(2)   
        closest_fire = tree.data[tree.query([boid.position], k=1,return_distance=False)[0,0]]
        force = closest_fire - boid.position
        return force    

=======
>>>>>>> 6f7875db449325a16130a9b193dbb96963538eef
    @staticmethod
    def crossover(rule: Rule, other: Rule) -> Rule:
        weight = rule.weight if np.random.random() > 0.5 else other.weight
        return Rule(weight=weight)

    def mutate(self) -> None:
        self.weight = np.random.normal(self.weight, self.std)

    def __str__(self):
        return f"weight = {self.weight:.4}"    


class Alignment(Rule):

    def __init__(self, **params):
        super().__init__(**params)
    
    def apply(self, swarm: Swarm,positions: np.ndarray[float], velocities: np.ndarray[float],neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)
        for i, n_idx in enumerate(neighbours_idx):
<<<<<<< HEAD
            n_idx = n_idx[n_idx!=i]
            if n_idx.size == 0:
                continue
            force_vector[i] = np.mean([b.velocity for b in swarm.boids[n_idx]])
=======
            if n_idx.size > 0:
                force_vector[i] = np.mean(velocities[n_idx,:],axis=0)
>>>>>>> 6f7875db449325a16130a9b193dbb96963538eef
        return force_vector

    @staticmethod
    def crossover(rule: Alignment, other: Alignment) -> Alignment:
        new_rule = super().crossover(rule, other)
        new_rule.__class__ = Alignment
    
        return new_rule
    
    def __str__(self):
        return f"Alignment: {super().__str__()}"

    


class Cohesion(Rule):

    def __init__(self, **params):
        super().__init__(**params)
    
    def apply(self, swarm: Swarm,positions: np.ndarray[float], velocities: np.ndarray[float],neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)
        for i, n_idx in enumerate(neighbours_idx):
<<<<<<< HEAD
            n_idx = n_idx[n_idx!=i]
            if n_idx.size == 0:
                continue
            force_vector[i] = np.mean([b.position for b in swarm.boids[n_idx]]) - swarm.boids[i].position
        force_vector = normalize(force_vector, axis=1) * self.strength
=======
            if n_idx.size > 0:
                force_vector[i] = np.mean(positions[n_idx,:],axis=0) - swarm.boids[i].position
        norm=np.linalg.norm(force_vector,axis=1)[:,np.newaxis]
        force_vector=np.divide(force_vector,norm,where=norm>0,out=force_vector)
>>>>>>> 6f7875db449325a16130a9b193dbb96963538eef
        return force_vector

    @staticmethod
    def crossover(rule: Cohesion, other: Cohesion) -> Cohesion:
        new_rule = super().crossover(rule,other)
        new_rule.__class__ = Cohesion

        return new_rule
    
    def __str__(self):
        return f"Cohesion: {super().__str__()}"


class Separation(Rule):

    def __init__(self, **params):
        super().__init__(**params)
    
    def apply(self, swarm: Swarm, positions: np.ndarray[float], velocities: np.ndarray[float],neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)
        for i, n_idx in enumerate(neighbours_idx):
<<<<<<< HEAD
            n_idx = n_idx[n_idx!=i]
            if n_idx.size == 0:
                continue
            force_vector[i] = swarm.boids[i].position - swarm.boids[n_idx[0]].position
        force_vector = normalize(force_vector, axis=1) * self.strength
=======
            if n_idx.size > 0:
                force_vector[i] = swarm.boids[i].position - swarm.boids[n_idx[0]].position 
        norm=np.linalg.norm(force_vector,axis=1)[:,np.newaxis]
        force_vector=np.divide(force_vector,norm,where=norm>0,out=force_vector) 

>>>>>>> 6f7875db449325a16130a9b193dbb96963538eef
        return force_vector

    @staticmethod
    def crossover(rule: Separation, other: Separation) -> Separation:
        new_rule = super().crossover(rule,other)
        new_rule.__class__ = Separation

        return new_rule

    def __str__(self):
        return f"Separation: {super().__str__()}"
    

class GoToWater(Rule):

    def __init__(self, **params):
        super().__init__(**params)
    
    def apply(self, swarm: Swarm, positions: np.ndarray[float], velocities: np.ndarray[float], neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)    
<<<<<<< HEAD
        
        for i, boid in enumerate(swarm.boids):
            if not boid.carrying_water:
                force_vector[i] = self.to_closest_tile(swarm.vision_range, swarm.env.water_tree, boid)
            else: 
                force_vector[i] = np.zeros(2)    
            force_vector = normalize(force_vector, axis=1) * self.strength
=======
        no_water_idx = np.nonzero([not b.carrying_water for b in swarm.boids])[0]
        if no_water_idx.size == 0: 
            return force_vector 
        water_within_vision_idx, _ = swarm.env.water_tree.query_radius(positions[no_water_idx], r=swarm.vision_range, sort_results=True, return_distance=True)

        for i,closest_waters in zip(no_water_idx,water_within_vision_idx):
            if closest_waters.size>0:
                force_vector[i]=swarm.env.water_tree.data[closest_waters[0]]-positions[i]
       
        norm=np.linalg.norm(force_vector,axis=1)[:,np.newaxis]
        force_vector=np.divide(force_vector,norm,where=norm>0,out=force_vector)
>>>>>>> 6f7875db449325a16130a9b193dbb96963538eef
        return force_vector

    @staticmethod
    def crossover(rule: GoToWater, other: GoToWater) -> GoToWater:
        new_rule = super().crossover(rule,other)
        new_rule.__class__ = GoToWater

        return new_rule

    def __str__(self):
        return f"GoToWater: {super().__str__()}"

class GoToFire(Rule):

    def __init__(self, **params):
        super().__init__(**params)
<<<<<<< HEAD
        self.strength = strength

    def apply(self, swarm: Swarm, velocities: np.ndarray[float], neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)   
        if swarm.env.n_fires ==0: 
            return force_vector
        
        fire_coordinates = []
        for i in range(swarm.env.grid.shape[0]):
            for j in range(swarm.env.grid.shape[1]):
                if swarm.env.grid[i, j] == State.FIRE.value:
                    fire_coordinates.append([i, j])

                    
        fire_tree=KDTree(fire_coordinates)
        for i, boid in enumerate(swarm.boids):
            if boid.carrying_water:
                force_vector[i] = self.to_closest_tile(swarm.vision_range, fire_tree, boid)
            else: 
                force_vector[i] = np.zeros(2)    
            force_vector = normalize(force_vector, axis=1) * self.strength
=======
    
    def apply(self, swarm: Swarm, positions: np.ndarray[float], velocities: np.ndarray[float], neighbours_idx:np.ndarray[np.ndarray]) -> np.ndarray[float]:
        force_vector = np.zeros(velocities.shape)        
        water_idx = np.nonzero([b.carrying_water for b in swarm.boids])[0]
        if swarm.env.n_fires == 0 or water_idx.size == 0: 
            return force_vector  
        fire_within_vision_idx, _ = swarm.env.fire_tree.query_radius(positions[water_idx], r=swarm.vision_range, sort_results=True, return_distance=True)

        for i,closest_fires in zip(water_idx,fire_within_vision_idx):
            if closest_fires.size>0:
                force_vector[i]=swarm.env.fire_tree.data[closest_fires[0]]-positions[i]
               
        norm=np.linalg.norm(force_vector,axis=1)[:,np.newaxis]
        force_vector=np.divide(force_vector,norm,where=norm>0,out=force_vector)
>>>>>>> 6f7875db449325a16130a9b193dbb96963538eef
        return force_vector

    @staticmethod
    def crossover(rule: GoToFire, other: GoToFire) -> GoToFire:
        new_rule = super().crossover(rule,other)
        new_rule.__class__ = GoToFire

        return new_rule

    def __str__(self):
        return f"GoToFire: {super().__str__()}"
