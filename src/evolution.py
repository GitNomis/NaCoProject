from __future__ import annotations
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from .environment import Environment # you're welcome :D

import numpy as np
from .import rule
from .swarm import Swarm 


class Evolution:
    population_size: int
    population: List[Swarm]
    environment: Environment
    generation: int

    def __init__(self, environment: Environment, population_size: int, mutate_rate:float):
        self.population_size = population_size
        self.mutate_rate = mutate_rate
        self.population = []
        for _ in range(population_size):
            rules = [rule.Alignment(weight=np.random.uniform(-1,3)), 
                     rule.Cohesion(weight=np.random.uniform(-1,3)), 
                     rule.Separation(weight=np.random.uniform(-1,3)), 
                     rule.GoToWater(weight=np.random.uniform(-1,3)), 
                     rule.GoToFire(weight=np.random.uniform(-1,3))]
            self.population.append(Swarm(environment.copy(),4,4,20,rules))
        self.environment = environment
        self.generation = 0

    def evolve(self, n_iters,reps=1) -> None:
        fitness = self.calculate_fitness(n_iters=n_iters,reps=reps)
        p = (fitness + 1 + abs(fitness.min()))**2
        candidates = np.random.choice(
            self.population, p=p/p.sum(), size=(self.population_size, 2), replace=True)
        new_generation = []
        for p1, p2 in candidates:
            child = self.crossover(p1, p2)
            self.mutate(child)
            new_generation.append(child)
        self.population = new_generation
        self.generation += 1

        return fitness

    def calculate_fitness(self, n_iters,reps=1) -> np.ndarray[int]:
        fitness = np.zeros(self.population_size)    
        for _ in range(reps):
            fitness += np.array([swarm.simulate(n_iters=n_iters) for swarm in self.population])
            for s in self.population:
                s.reset(self.environment.copy())
        return  fitness/reps

    def crossover(self, parent1: Swarm, parent2: Swarm) -> Swarm:
        rule_types = parent1.rules.keys() | parent2.rules.keys()
        new_rules = [rule.crossover(parent1.rules[rule], parent2.rules[rule]) for rule in rule_types]
        return Swarm(self.environment.copy(), parent1.vision_range, parent2.vision_range, len(self.population), new_rules)

    def mutate(self, child: Swarm) -> None:
        for rule in child.rules.keys():
            if np.random.random()<=self.mutate_rate: 
                child.rules[rule].mutate()
