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

    def __init__(self, environment: Environment, population_size: int):
        self.population_size = population_size
        rules = [rule.Alignment(weight=0.3) , rule.Cohesion(weight=0.5), rule.Separation(weight=0.4, strength=1.5), rule.GoToWater(), rule.GoToFire()]
        self.population = [Swarm(environment.copy(), 4,4, 20, rules)
                           for _ in range(population_size)]
        self.environment = environment
        self.generation = 0

    def evolve(self, n_iters) -> None:
        fitness = self.calculate_fitness(n_iters=n_iters)
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

    def calculate_fitness(self, n_iters) -> np.ndarray[int]:
        return  np.array([swarm.simulate(n_iters=n_iters) for swarm in self.population])

    def crossover(self, parent1: Swarm, parent2: Swarm) -> Swarm:
        rule_types = parent1.rules.keys() | parent2.rules.keys()
        new_rules = [rule.crossover(parent1.rules[rule], parent2.rules[rule]) for rule in rule_types]
        return Swarm(self.environment.copy(), 4,4, 20, new_rules)

    def mutate(self, child: Swarm) -> None:
        for rule in child.rules.keys():
            if np.random.random()<=0.1: 
                child.rules[rule].mutate()
