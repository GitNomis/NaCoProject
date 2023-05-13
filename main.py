from src import Environment, Display
from src.Swarm import Swarm

import itertools

def main():
    show_display=True
    steps=50

    env = Environment.Environment(size=(5, 5), swarm_size=10, fire_size=0.5)
    display = Display.Display(env, steps=steps)

    if show_display:
        display.display()
    else:
        for _ in itertools.repeat(None, steps):
            env.update()
        print(env.grid)

if __name__ == '__main__':
    main()