import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters for the simulation."""
    grid_size: int = 100
    initial_density: float = 0.2
    rebirth_threshold: float = 0.7
    death_threshold: float = 0.3
    mutation_rate: float = 0.05
    history_length: int = 0  # Track simulation history.  0 means no history.


class Grid:
    """Represents the cellular automaton grid."""
    def __init__(self, size: int, initial_density: float):
        self.size = size
        self.grid = np.random.rand(size, size) < initial_density

    def update(self, config: Config):
        """Updates the grid state based on the given configuration."""
        kernel = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])  # Sum neighbors, excluding center
        density = convolve(self.grid, kernel, mode='same')

        new_grid = self.grid.copy()
        new_grid[density > config.rebirth_threshold] = True
        new_grid[density < config.death_threshold] = False

        # Mutation: Flip state of a small percentage of alive cells
        alive_cells = np.sum(self.grid)
        num_mutations = int(config.mutation_rate * alive_cells)
        if num_mutations > 0:
            mutation_indices = np.random.choice(np.where(self.grid)[0], num_mutations, replace=False)
            self.grid[mutation_indices] = ~self.grid[mutation_indices] # Flip the cells

        self.grid = new_grid

    def get_grid(self):
        """Returns a copy of the grid."""
        return self.grid.copy() # Return a copy to prevent external modification


class Simulator:
    """Orchestrates the simulation process."""
    def __init__(self, grid: Grid, config: Config):
        self.grid = grid
        self.config = config

    def run_step(self):
        """Runs a single simulation step."""
        self.grid.update(self.config)

    def get_grid(self):
        """Returns a copy of the grid."""
        return self.grid.get_grid()


class Visualizer:
    """Handles the visualization of the grid."""
    def __init__(self, grid: np.ndarray, interval: int = 100):
        self.grid = grid
        self.interval = interval

    def animate(self, simulator: Simulator):
        """Animates the simulation."""
        fig, ax = plt.subplots()
        img = ax.imshow(simulator.get_grid(), cmap='viridis', interpolation='nearest')
        ax.set_title("Cellular Automata")

        def update_fig(*args):
            simulator.run_step()
            img.set_array(simulator.get_grid())
            return img,

        ani = animation.FuncAnimation(fig, update_fig, interval=self.interval, blit=True, repeat=True)
        plt.show()


# --- Main Execution ---

# Configuration
config = Config(history_length=0)  # No history tracking for now

# Initialize Grid and Simulator
grid = Grid(config.grid_size, config.initial_density)
simulator = Simulator(grid, config)

# Initialize Visualizer
visualizer = Visualizer(grid.grid, interval=100)

# Run the animation
visualizer.animate(simulator)
