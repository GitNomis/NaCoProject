from src import Environment, Display
from src.Swarm import Swarm

def main():
    env = Environment.Environment(size=(5, 5), swarm_size=10, fire_size=0.5)
    print(env.grid)
    print(Display.display(env))
    for iter in range(10):
        env.update()


if __name__ == '__main__':
    main()