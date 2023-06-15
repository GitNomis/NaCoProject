import numpy as np
from tqdm import tqdm
from src import *

def main():
    results_file = r"./output.txt"
    show_display=True
    n_observations = 1000
    steps=500  # How long the simulation after the evolution should run
    infinite=False
    n_iters = 200  # How long the simultations during the evolution should run
    key_order = [rule.Alignment, rule.Cohesion, rule.Separation, rule.GoToFire, rule.GoToWater]

    #env = Environment.example(size=(20, 20), fire_size=10, water_size=2)
    env = Environment.from_file(r'grid_files\presentation.in')

    with open(results_file, mode='a') as out:
        out.write("Obs,Gen,Alignment,Cohesion,Separation,GoToFire,GoToWater,fitness\n")

    for obs in tqdm(range(n_observations), position=0):
        evo = Evolution(env,40,0.05)
        for g in tqdm(range(50), position=1, leave=None):
            fitness = evo.evolve(n_iters=n_iters, reps=2)
            
            with open(results_file, mode='a') as out:
                for i,s in enumerate(evo.population):
                    out.write("{:>3}, {:>3}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>6.2f}\n".format(obs,evo.generation-1,*[s.rules[k].weight for k in key_order],fitness[i]))

        fitness = evo.calculate_fitness(n_iters=n_iters,reps=5)
        swarm = evo.population[np.argmax(fitness)]
        swarm = Swarm(env.copy(), swarm.vision_range, swarm.max_speed, swarm.boids.size, swarm.rules.values())
        with open(results_file, mode='a') as out:
            for i,s in enumerate(evo.population):
                out.write("{:>3}, {:>3}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>6.2f}\n".format(obs,evo.generation-1,*[s.rules[k].weight for k in key_order],fitness[i]))

    display = Display(swarm, steps=steps,infinite=infinite)

    if show_display:
        display.display()
    else:
        for _ in range(steps):
            swarm.update()
        print(env.grid) 
    

if __name__ == '__main__':
    main()
