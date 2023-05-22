from src import *

def main():
    show_display=True
    steps=300
    infinite=False

    env = Environment.example(size=(10, 10), fire_size=1, water_size=1)
    evo = Evolution(env,10)

    while evo.generation < 10:  
        evo.simulate(n_iters=100)
        fitness = evo.evolve()
        print(f"Gen {evo.generation:>3}: {max(fitness)}")

    swarm = evo.population[0]

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
    
    evo.evolve()    
    

if __name__ == '__main__':
    main()
