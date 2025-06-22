import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
GRID_SIZE = 100
INITIAL_CELL_PROBABILITY = 0.1
REPRODUCTION_THRESHOLD = 6
DEATH_THRESHOLD = 2
ADAPTATION_RATE = 0.02
TOROIDAL_BOUNDARY = True
MAX_NEIGHBORS = 8  # Number of neighbors considered for each cell.  Crucial for adaptation.

# Initialize grid randomly
grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[1-INITIAL_CELL_PROBABILITY, INITIAL_CELL_PROBABILITY])

# Pre-calculate neighbor counts
neighbor_counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int_)


def calculate_neighbors(grid, i, j, toroidal):
    """Calculates the number of live neighbors for a given cell.

    Args:
        grid: The grid array.
        i: The row index of the cell.
        j: The column index of the cell.
        toroidal: Whether to use toroidal boundaries.

    Returns:
        The number of live neighbors.
    """
    neighbors = 0
    for x in range(max(0, i - 1), min(GRID_SIZE, i + 2)):
        for y in range(max(0, j - 1), min(GRID_SIZE, j + 2)):
            if (x, y) != (i, j):
                neighbors += grid[x, y]
    return neighbors


def update_grid(grid, toroidal):
    """Updates the grid based on the cellular automata rules.

    Args:
        grid: The grid array.
        toroidal: Whether to use toroidal boundaries.

    Returns:
        The updated grid array.
    """
    new_grid = np.copy(grid)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            neighbors = calculate_neighbors(grid, i, j, toroidal)

            # Reproduction rule: Cell reproduces if alive and has more than REPRODUCTION_THRESHOLD neighbors.
            if grid[i, j] == 1 and neighbors > REPRODUCTION_THRESHOLD:
                new_grid[i, j] = 1
            # Death rule: Cell dies if alive and has fewer than DEATH_THRESHOLD neighbors.
            elif grid[i, j] == 1 and neighbors < DEATH_THRESHOLD:
                new_grid[i, j] = 0
            # Adaptation rule: A dead cell becomes alive with probability ADAPTATION_RATE * (neighbors / MAX_NEIGHBORS).
            elif grid[i, j] == 0 and neighbors > 0:
                if np.random.rand() < ADAPTATION_RATE * (neighbors / MAX_NEIGHBORS):
                    new_grid[i, j] = 1

    return new_grid


# Visualization setup
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='viridis', interpolation='nearest')

def animate(frame):
    """Animation function to update the grid and display it."""
    global grid
    grid = update_grid(grid, TOROIDAL_BOUNDARY)
    img.set_array(grid)
    return img,


# Create animation
ani = animation.FuncAnimation(fig, animate, interval=50, repeat=True)

plt.show()
