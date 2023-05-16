from __future__ import annotations
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    pass  # you're welcome :D

import numpy as np
from .environment import Environment    

class Evolution:
    population_size:int
    population:List[Environment]
    generation:int

    def __init__(self,population_size:int) -> None:
        population_size=population_size
        population=[Environment((50,50),20,1,1)for _ in range(population_size)]
        generation=0

    def evolve(self) -> None: 
        fitness = np.array(env.calculate_fitness() for env in self.population)
        candidates=np.random.choice(self.population,p=fitness/fitness.sum(),size=(self.population_size,2),replace=True)
        new_generation = []
        for p1,p2 in candidates:
            child=self.crossover(p1,p2)
            self.mutate(child)
            new_generation.append(child)
        self.population=new_generation
        self.generation+=1


    def crossover(self,parent1:Environment,parent2:Environment)->Environment:
        pass        

    def mutate(self,child:Environment)->None:
        pass
