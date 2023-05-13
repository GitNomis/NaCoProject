from .Environment import Environment
from .State import State

import matplotlib.pyplot as plt
from matplotlib import colors, animation
import seaborn as sns

class Display:

    def __init__(self, env:Environment, steps:int) -> None:
        self.env = env
        self.steps = steps

        self.cmap = colors.ListedColormap(['moccasin','firebrick','deepskyblue','yellow'])
        bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        self.interval = 100
        self.boid_marker = ">"

    def display(self):
        fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.ax.set_xlim([0,self.env.grid.shape[0]])
        self.ax.set_ylim([0,self.env.grid.shape[1]])

        # Forest fire heatmap
        sns.heatmap(self.env.grid, cbar=False, cmap=self.cmap, norm=self.norm)
        
        # Boids scatterplot
        x = [boid.position[0] for boid in self.env.swarm.boids]
        y = [boid.position[1] for boid in self.env.swarm.boids]
        self.scatter = plt.scatter(x, y, marker=self.boid_marker)

        # Animation
        animator = animation.FuncAnimation(fig, self.animate, interval=self.interval, frames=self.steps, repeat=False, cache_frame_data=False)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(self.steps * self.interval * 0.001) # pause (s) = frames * interval (ms) * 0.001
        plt.close(fig)
        
    def animate(self, i:int):
        # Clear axes
        self.ax.cla()
        
        # Redraw forest fire heatmap
        sns.heatmap(ax=self.ax, data=self.env.grid, cbar=False, cmap=self.cmap, norm=self.norm)
        
        # Redraw boids scatterplot
        x = [boid.position[0] for boid in self.env.swarm.boids]
        y = [boid.position[1] for boid in self.env.swarm.boids]
        self.scatter = plt.scatter(x, y, marker=self.boid_marker)
        
        # Update the boids
        offsets, markers = self.env.update()
        self.scatter.set_offsets(offsets)
        paths = [m.get_path().transformed(m.get_transform()) for m in markers]
        self.scatter.set_paths(paths)

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