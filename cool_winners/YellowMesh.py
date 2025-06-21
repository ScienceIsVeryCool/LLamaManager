import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

def update(grid):
    size = grid.shape[0]
    new_grid = np.copy(grid)

    for i in range(size):
        for j in range(size):
            n = np.sum(grid[(i-1)%size:(i+2)%size, (j-1)%size:(j+2)%size]) - grid[i, j]

            # Basic rules (Game of Life) - Modified Thresholds
            if grid[i, j] == 1 and (n < 2 or n > 4):  # Slightly loosened survival conditions
                new_grid[i, j] = 0
            elif grid[i, j] == 0 and n == 2:
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and n == 3:
                new_grid[i, j] = 1

    return new_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellular Automata Simulation")
    parser.add_argument("--size", type=int, default=100, help="Size of the grid")
    parser.add_argument("--percentage", type=float, default=0.3, help="Initial cell percentage")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    grid = np.random.choice([0, 1], size=(args.size, args.size), p=[1 - args.percentage, args.percentage])

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='viridis', animated=True)

    def init():
        im.set_data(grid)
        return im,

    def animate(frame):
        global grid
        grid = update(grid)
        im.set_data(grid)
        return im,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=args.gens, repeat=False)
    plt.show()