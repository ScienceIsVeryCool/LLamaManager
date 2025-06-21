import pygame
import numpy as np
import math
import random
# --- Constants ---
GRID_WIDTH = 500
GRID_HEIGHT = 500
CELL_SIZE = 2
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

SIGNAL_DIFFUSION_RATE = 0.2
SIGNAL_ATTRACTION_RATE = 0.08
SIGNAL_DECAY_RATE = 0.97
SIGMOID_STEEPNESS = 2.5
EMITTER_PROBABILITY = 0.3
SIGNAL_EMISSION_AMOUNT = 0.3
SIGNAL_ABSORPTION_AMOUNT = 0.12
BURST_CHANCE = 0.02
BURST_STRENGTH = 0.7
SPECIAL_SIGNAL_CHANCE = 0.03
SPECIAL_SIGNAL_RANGE = 0.7
SPECIAL_SIGNAL_DECAY = 0.9
DRIFT_CHANCE = 0.015
MUTATION_CHANCE = 0.002
CELL_MEMORY_SIZE = 3
MOOD_STRENGTH = 0.15
MOOD_CHANGE_RATE = 0.01
SIGNAL_COLOR_SCALE = 255
COLOR_SHIFT_SPEED = 0.005
INTERACTION_THRESHOLD = 0.6


# --- Color Definitions ---
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

# --- Helper Functions ---
def sigmoid(x, steepness=SIGMOID_STEEPNESS):
    return 1 / (1 + np.exp(-steepness * x))


class CellularAutomaton:
    def __init__(self, width, height, initial_prob=0.5):
        self.width = width
        self.height = height
        self.grid = np.random.choice([0, 1], size=(height, width), p=[1 - initial_prob, initial_prob])
        self.signal_field = np.zeros((height, width), dtype=np.float32)
        self.mood = 0.0
        self.color_palette = self.generate_color_palette()
        self.color_offset = 0.0

    def generate_color_palette(self):
        """Generates a rainbow-like color palette."""
        palette = []
        for i in range(256):
            r = int((1 + math.sin(i * 0.1)) * 127)
            g = int((1 + math.cos(i * 0.15)) * 127)
            b = int((1 + math.sin(i * 0.2)) * 127)
            palette.append((r, g, b))
        return palette

    def update_mood(self):
        avg_signal = np.mean(self.signal_field)
        self.mood += (avg_signal - 0.5) * MOOD_CHANGE_RATE
        self.mood = np.clip(self.mood, -1, 1)

    def diffuse_signal(self):
        """Diffuses the signal field using a simple iterative update."""
        new_signal_field = np.copy(self.signal_field)  # Create a copy for updates
        for x in range(self.width):
            for y in range(self.height):
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if (nx, ny) != (x, y): # Avoid self-interaction
                                new_signal_field[x, y] += self.signal_field[nx, ny] * SIGNAL_DIFFUSION_RATE
        self.signal_field = new_signal_field

    def update(self):
        self.update_mood()
        new_signal_field = np.zeros_like(self.signal_field)

        # Dynamic Parameters based on Mood
        emission_amount = SIGNAL_EMISSION_AMOUNT + self.mood * 0.2
        absorption_amount = SIGNAL_ABSORPTION_AMOUNT * (1 - self.mood * 0.2)

        # Update signal emission/absorption
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == 1:
                    new_signal_field[x, y] += emission_amount
                else:
                    new_signal_field[x, y] -= absorption_amount

        # Signal Decay
        new_signal_field = new_signal_field * SIGNAL_DECAY_RATE

        # Grid State Transitions: Probability based on signal and mood
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < sigmoid(new_signal_field[x, y] + self.mood, 1.0):
                    self.grid[x, y] = 1 - self.grid[x, y]

        self.signal_field = new_signal_field

        # Color Palette Shift
        self.color_offset += COLOR_SHIFT_SPEED
        self.color_palette = self.shift_color_palette(self.color_palette, self.color_offset)

    def shift_color_palette(self, palette, offset):
        """Shifts the color palette by a given offset."""
        shifted_palette = np.roll(palette, int(offset) % len(palette))
        return shifted_palette


# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Cellular Automaton")
clock = pygame.time.Clock()

# --- Main Loop ---
ca = CellularAutomaton(GRID_WIDTH, GRID_HEIGHT)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ca.update()

    # Render the grid
    screen.fill(BLACK)
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if ca.grid[y, x] == 1:
                color = ca.color_palette[ (y * GRID_WIDTH + x) % len(ca.color_palette)]
                pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)  # Adjust frame rate as needed

pygame.quit()
