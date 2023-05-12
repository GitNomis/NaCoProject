from .Environment import Environment
from .State import State

import matplotlib.pyplot as plt
from matplotlib import colors, animation
import seaborn as sns

class Display:

    def __init__(self, env:Environment, steps:int) -> None:
        self.env = env
        self.steps = steps

        self.cmap = colors.ListedColormap(['moccasin','firebrick','green','yellow'])
        bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        self.interval = 200

    def display(self):

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.tight_layout()

        sns.heatmap(self.env.grid, cbar=False, cmap=self.cmap, norm=self.norm)

        animator = animation.FuncAnimation(fig, self.animate, interval=self.interval, frames=self.steps, repeat=False, cache_frame_data=False)

        plt.show(block=False)
        plt.pause(self.steps * self.interval * 0.001) # pause (s) = frames * interval (ms) * 0.001
        plt.close("all")
        
    def animate(self, i:int):
        self.env.update()
        sns.heatmap(self.env.grid, cbar=False, cmap=self.cmap, norm=self.norm) # Update the fire
        # TODO: Update the boids

    def display_grid(self, env:Environment):
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