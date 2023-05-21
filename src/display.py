from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .environment import Environment
    
import matplotlib.pyplot as plt
from matplotlib import colors, animation
from matplotlib.markers import MarkerStyle
import numpy as np
import seaborn as sns
from .state import State



class Display:

    def __init__(self, env:Environment, steps:int) -> None:
        """Display class with different ways of displaying an environment.

        Args:
            env (Environment): environment to display
            steps (int): number of steps
        """
        self.env = env
        self.steps = steps

        self.cmap = colors.ListedColormap(['moccasin','firebrick','deepskyblue','yellow'])
        bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        self.interval = 100
        self.boid_marker = ">"

    def display(self)-> None:
        fig, self.ax = plt.subplots(1, 1, figsize=(self.env.grid.shape[0], self.env.grid.shape[1]))
        self.ax.set_xlim([0,self.env.grid.shape[0]])
        self.ax.set_ylim([0,self.env.grid.shape[1]])

        # Forest fire heatmap
        sns.heatmap(self.env.grid.T, cbar=False, cmap=self.cmap, norm=self.norm)
        
        # Boids scatterplot
        x = [boid.position[0] for boid in self.env.swarm.boids]
        y = [boid.position[1] for boid in self.env.swarm.boids]
        self.scatter = plt.scatter(x, y, marker=self.boid_marker, color="grey")

        # Animation
        animator = animation.FuncAnimation(fig, self.animate, interval=self.interval, frames=self.steps, repeat=False, cache_frame_data=False)

        self.ax.invert_yaxis()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(self.steps * self.interval * 0.001) # pause (s) = frames * interval (ms) * 0.001
        plt.close(fig)
        
    def animate(self, i:int)-> None:
        # Clear axes
        self.ax.cla()
        
        # Redraw forest fire heatmap
        sns.heatmap(ax=self.ax, data=self.env.grid.T, cbar=False, cmap=self.cmap, norm=self.norm)
        
        # Redraw boids scatterplot
        x = [boid.position[0] for boid in self.env.swarm.boids]
        y = [boid.position[1] for boid in self.env.swarm.boids]
        self.scatter = plt.scatter(x, y, marker=self.boid_marker, color="grey")
        
        # Update the boids
        self.env.update()

        # TODO: Add updating direction for plotting
        offsets = []
        markers = []
        colors = []
        
        for boid in self.env.swarm.boids:
            offsets.append([boid.position[0], boid.position[1]])
            marker = MarkerStyle(">")
            marker._transform = marker.get_transform().rotate_deg(np.angle(complex(*boid.velocity),True))
            marker._transform = marker.get_transform().scale(2, 2)
            if boid.carrying_water:
                colors.append("blue")
            else:
                colors.append("grey")
            markers.append(marker)
            
        self.scatter.set_color(colors)
        
        self.scatter.set_offsets(offsets)
        paths = [m.get_path().transformed(m.get_transform()) for m in markers]
        self.scatter.set_paths(paths)

    def display_grid(self, env:Environment)-> None:
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