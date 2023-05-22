from src import *

import itertools

def main():
    show_display=True
    steps=100
    infinite=True

    env = Environment(size=(5, 5), swarm_size=5, fire_size=1, water_size=1)
    display = Display(env, steps=steps, infinite=infinite)

    if show_display:
        display.display()
    else:
        for _ in itertools.repeat(None, steps):
            env.update()
        print(env.grid)

if __name__ == '__main__':
    main()