import numpy as np
from src import *

def main():
    show_display=True
    steps=300
    infinite=False
    n_iters = 100

    env = Environment.example(size=(15, 15), fire_size=10, water_size=1)
    evo = Evolution(env,10)
    while evo.generation < 10: 
        fitness = evo.evolve(n_iters=n_iters)
        print(f"Gen {evo.generation-1:>3}: {max(fitness)}")

    fitness = evo.calculate_fitness(n_iters=n_iters)
    swarm = evo.population[np.argmax(fitness)]
    swarm.env = env.copy()
    for b in swarm.boids:
        b.env=swarm.env
        b.carrying_water=False
    #swarm = Swarm(env,2,20,[rule.Alignment(weight=0.2),rule.Cohesion(strength=2,weight=0.4),rule.Separation(strength=1,weight=0.4)])
    print("Rules:",*swarm.rules.values())
    display = Display(swarm, steps=steps,infinite=infinite)

    if show_display:
        display.display()
    else:
        for _ in range(steps):
            swarm.update()
        print(env.grid)

    env = Environment.example(size=(5, 5), fire_size=1, water_size=1)
    
    evo.evolve(n_iters = 100)    
    

if __name__ == '__main__':
    main()
