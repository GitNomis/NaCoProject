import numpy as np
from src import *

def main():
    show_display=True
    steps=300
    infinite=False
    n_iters = 100

    env = Environment.example(size=(20, 20), fire_size=10, water_size=2)
    evo = Evolution(env,10)
    while evo.generation < 10: 
        fitness = evo.evolve(n_iters=n_iters)
        print(f"Gen {evo.generation-1:>3}: Avg: {np.mean(np.sqrt(fitness))}, Max: {np.max(np.sqrt(fitness))}")

    fitness = evo.calculate_fitness(n_iters=n_iters)
    swarm = evo.population[np.argmax(fitness)]
    swarm.env = env.copy()
    for b in swarm.boids:
        b.env=swarm.env
        b.carrying_water=False
    #swarm = Swarm(env,4,4,20,[rule.Alignment(weight=-0.2),rule.Cohesion(strength=0.5,weight=0.5),rule.Separation(strength=1.5,weight=0.5),rule.GoToFire(weight=2.0,strength=2.0),rule.GoToWater(weight=2.0,strength=2.0)])
    print("Rules:",*swarm.rules.values())
    display = Display(swarm, steps=steps,infinite=infinite)

    if show_display:
        display.display()
    else:
        for _ in range(steps):
            swarm.update()
        print(env.grid)

    #env = Environment.example(size=(5, 5), fire_size=1, water_size=1)
    
    #evo.evolve(n_iters = 100)    
    

if __name__ == '__main__':
    main()
