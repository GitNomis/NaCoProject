from .Environment import Environment
from .State import State

def display(env:Environment):
    shape=env.grid.shape
    res = ""
    for y in range(shape[0]):
        for x in range(shape[1]):
            if env.grid[y,x]==State.BARREN:
                res+= '🏜'
            elif env.grid[y,x]==State.FIRE:
                res+= '🔥'
            elif env.grid[y,x]==State.TREE:
                res+= '🌲'
            elif env.grid[y,x]==State.WATER:
                res+= '🌊'
        res+='\n'
    return res    