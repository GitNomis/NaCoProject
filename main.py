from src import Environment, Display

def main():
    env = Environment.Environment(size=(5, 5), swarm_size=10, fire_size=0.5)
    print(env.grid)
    print(Display.display(env))

if __name__ == '__main__':
    main()