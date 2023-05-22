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
        rules = [rule.Alignment(), rule.Cohesion(), rule.Separation()]
        self.population = [Swarm(environment.copy(), 10, 20, rules)
                           for _ in range(population_size)]
        self.environment = environment
        self.generation = 0

    def evolve(self) -> None:
        fitness = np.array([swarm.env.calculate_fitness()
                           for swarm in self.population])
        candidates = np.random.choice(
            self.population, p=fitness/fitness.sum(), size=(self.population_size, 2), replace=True)
        new_generation = []
        for p1, p2 in candidates:
            child = self.crossover(p1, p2)
            self.mutate(child)
            new_generation.append(child)
        self.population = new_generation
        self.generation += 1

        return fitness
    
    def simulate(self,n_iters):
        for _ in range(n_iters):
            for s in self.population:
                s.update()

    def crossover(self, parent1: Swarm, parent2: Swarm) -> Swarm:
        rule_types = parent1.rules.keys() | parent2.rules.keys()
        new_rules = [rule.crossover(parent1.rules[rule], parent2.rules[rule]) for rule in rule_types]
        return Swarm(self.environment.copy(), 10, 20, new_rules)

    def mutate(self, child: Swarm) -> None:
        for rule in child.rules.keys():
            child.rules[rule].mutate()
