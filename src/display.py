from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .swarm import Swarm
    from .environment import Environment
    
import matplotlib.pyplot as plt
from matplotlib import colors, animation
from matplotlib.markers import MarkerStyle
import numpy as np
import seaborn as sns
from .state import State



class Display:

    def __init__(self, swarm:Swarm, steps:int, infinite:bool,savefile:str=None) -> None:
        """Display class with different ways of displaying an environment.

        Args:
            env (Environment): environment to display
            steps (int): number of simulation steps
            infinite (bool): whether the simulation runs infinitely or not
        """
        self.swarm = swarm
        self.steps = steps
        self.infinite = infinite
        self.savefile = savefile

        self.cmap = colors.ListedColormap(['peru','firebrick','dodgerblue','forestgreen'])
        bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        self.interval = 100
        self.boid_marker = ">"
        self.boid_color = ["darkgray","deepskyblue"]

    def display(self)-> None:
        """Show and animate the display
        """        
        fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        # Forest fire
        plt.imshow(np.flip(self.swarm.env.grid.T, 0), cmap=self.cmap, norm=self.norm, origin="lower", extent=(0, self.swarm.env.grid.shape[0], self.swarm.env.grid.shape[1], 0))
        
        # Boids scatterplot
        x = [boid.position[0] for boid in self.swarm.boids]
        y = [boid.position[1] for boid in self.swarm.boids]
        self.scatter = plt.scatter(x, y, marker=self.boid_marker, color=self.boid_color[0])

        self.ax.invert_yaxis()
        if self.savefile:
            self.ax.axis('off')
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        
        if not self.infinite:
            # Animation
            animator = animation.FuncAnimation(fig, self.animate, interval=self.interval, frames=self.steps, repeat=False, cache_frame_data=False)
            if self.savefile:
                animator.save(f'{self.savefile}.gif',fps=60,dpi=300)
            else:
                plt.show(block=False)
                plt.pause(self.steps * self.interval * 0.001) # pause (s) = frames * interval (ms) * 0.001    
            plt.close(fig)
        else:
            # Animation
            animator = animation.FuncAnimation(fig, self.animate, interval=self.interval, cache_frame_data=False)
            plt.show()
        
    def animate(self, i:int)-> None:
        """Animate the display

        Args:
            i: not accessed.
        """        
        # Update the boids
        self.swarm.update()
        
        # Clear axes & set limits
        self.ax.cla()
        self.ax.set_xlim([0,self.swarm.env.grid.shape[0]])
        self.ax.set_ylim([0,self.swarm.env.grid.shape[1]])
        
        # Redraw forest fire
        plt.imshow(np.flip(self.swarm.env.grid.T, 0), cmap=self.cmap, norm=self.norm, origin="lower", extent=(0, self.swarm.env.grid.shape[0], self.swarm.env.grid.shape[1], 0))
        
        # Redraw boids scatterplot
        x = [boid.position[0] for boid in self.swarm.boids]
        y = [boid.position[1] for boid in self.swarm.boids]
        self.scatter = plt.scatter(x, y, marker=self.boid_marker, color=self.boid_color[0])

        offsets = []
        markers = []
        colors = []
        
        for boid in self.swarm.boids:
            offsets.append([boid.position[0], boid.position[1]])
            marker = MarkerStyle(">")
            marker._transform = marker.get_transform().rotate_deg(np.angle(complex(*boid.velocity),True))
            marker._transform = marker.get_transform().scale(2, 2)
            if boid.carrying_water:
                colors.append(self.boid_color[1])
            else:
                colors.append(self.boid_color[0])
            markers.append(marker)
            
        self.scatter.set_color(colors)
        self.scatter.set_offsets(offsets)
        paths = [m.get_path().transformed(m.get_transform()) for m in markers]
        self.scatter.set_paths(paths)