import pygame
import random
import math
import numpy as np
import cProfile
import pstats
import time

# Constants
WIDTH = 800
HEIGHT = 600
FOOD_DENSITY = 0.05
MUTATION_RATE = 0.02
FPS = 60
CREATURE_SIZE = 15
FOOD_SIZE = 5
RESOURCE_AVAILABILITY = 100  # Global resource availability
POPULATION_SIZE = 10  # Initial population size - controllable via UI

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


class Creature:
    """
    Represents a creature in the simulation.
    """

    def __init__(self, x, y, genome=""):
        """
        Initializes a new creature.
        """
        if not genome:
            self.genome = ''.join(random.choice('01') for _ in range(10))
        else:
            self.genome = genome
        self.energy = 50
        self.size = self.decode_size(self.genome)
        self.speed = self.decode_speed(self.genome)
        self.x = x
        self.y = y
        self.dirty = False  # New attribute for dirty flag
        print(f"Creature created at ({self.x}, {self.y}) with genome: {self.genome}")

    def update(self, environment):
        """
        Updates the creature's state.
        """
        self.move_towards_food(environment)
        self.consume_food(environment)
        self.energy -= 0.5
        if self.energy <= 0:
            self.die()

    def move_towards_food(self, environment):
        """
        Moves the creature towards the closest food source.
        """
        food_positions = np.array(environment.food_sources)
        distances = np.linalg.norm(food_positions - np.array([self.x, self.y]), axis=1)
        closest_food_index = np.argmin(distances)
        closest_food = environment.food_sources[closest_food_index]

        angle = math.atan2(closest_food[1] - self.y, closest_food[0] - self.x)
        self.x += self.speed * math.cos(angle)
        self.y += self.speed * math.sin(angle)

        # Keep creatures within bounds
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))
        self.dirty = True  # Mark as dirty after movement

    def consume_food(self, environment):
        """
        Consumes food if available.  Optimized food consumption.
        """
        min_distance = float('inf')
        food_index = -1

        for i, food in enumerate(environment.food_sources):
            distance = math.sqrt((self.x - food[0])**2 + (self.y - food[1])**2)
            if distance < min_distance:
                min_distance = distance
                food_index = i

        if food_index != -1 and min_distance < self.size + FOOD_SIZE:
            self.energy += 10  # Increased energy gain
            environment.food_sources.pop(food_index)  # Remove food directly
            if len(environment.food_sources) < int(WIDTH * HEIGHT * FOOD_DENSITY * 0.5):  # Regenerate if too few
                environment.generate_food()
            self.dirty = True  # Mark as dirty after food consumption

    def reproduce(self, mate):
        """
        Handles creature reproduction.
        """
        new_genome = ""
        for i in range(10):
            if random.random() < MUTATION_RATE:
                new_genome += '1' if self.genome[i] == '0' else '0'
            else:
                new_genome += self.genome[i]

        new_creature = Creature(random.randint(0, WIDTH), random.randint(0, HEIGHT), new_genome)
        new_creature.energy = 10
        print(f"New creature created with genome: {new_creature.genome}")
        return new_creature

    def die(self):
        """
        Kills the creature.
        """
        print(f"Creature died at ({self.x}, {self.y}) with genome: {self.genome}")

    def decode_size(self, genome):
        """
        Decodes the creature's size from its genome.
        """
        return int(genome[:2], 2) + 5

    def decode_speed(self, genome):
        """
        Decodes the creature's speed from its genome.
        """
        return int(genome[2:4], 2) + 1


class Environment:
    """
    Represents the simulation environment.
    """

    def __init__(self, width, height):
        """
        Initializes the environment.
        """
        self.width = width
        self.height = height
        self.food_sources = []
        self.generate_food()

    def generate_food(self):
        """
        Generates food sources in the environment.
        """
        self.food_sources = []
        for _ in range(int(WIDTH * HEIGHT * FOOD_DENSITY)):
            self.food_sources.append((random.randint(0, WIDTH), random.randint(0, HEIGHT)))

    def update(self):
        """
        Updates the environment.
        """
        pass  # No more environment updates needed


class Simulation:
    """
    Manages the overall simulation.
    """

    def __init__(self, width, height, initial_population_size):
        self.environment = Environment(width, height)
        self.creatures = []
        for _ in range(initial_population_size):
            self.creatures.append(Creature(random.randint(0, width), random.randint(0, height)))

    def update(self):
        """
        Updates the simulation state.
        """
        self.reproduce()
        for creature in self.creatures:
            creature.update(self.environment)
        self.remove_dead_creatures()

    def reproduce(self):
        """
        Handles creature reproduction.
        """
        for creature in self.creatures:
            if creature.energy > 10 and random.random() < 0.02:  # Reduced reproduction rate
                mate = random.choice(self.creatures)
                if mate != creature:
                    new_creature = creature.reproduce(mate)
                    self.creatures.append(new_creature)

    def remove_dead_creatures(self):
        """
        Removes creatures that have died.
        """
        self.creatures = [creature for creature in self.creatures if creature.energy > 0]

    def draw(self, screen):
        """
        Draws the simulation state on the screen.
        """
        screen.fill((0, 0, 0))

        # Draw food
        for food in self.environment.food_sources:
            pygame.draw.circle(screen, GREEN, food, FOOD_SIZE)

        # Draw creatures
        for creature in self.creatures:
            if creature.dirty:
                pygame.draw.circle(screen, WHITE, (int(creature.x), int(creature.y)), creature.size)
                creature.dirty = False  # Reset dirty flag

        pygame.display.flip()


# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Genesis Engine - Core Simulation")
clock = pygame.time.Clock()

# Simulation setup
simulation = Simulation(WIDTH, HEIGHT, POPULATION_SIZE)

# Main loop
running = True
start_time = time.time()
while running:
    start_iter_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                new_population_size = int(input("Enter new population size: "))
                simulation.creatures = []
                for _ in range(new_population_size):
                    simulation.creatures.append(
                        Creature(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

    simulation.update()
    simulation.draw(screen)

    end_iter_time = time.time()
    clock.tick(FPS)

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

# cProfile output
# cProfile.run('main()', 'output.prof')
# p = pstats.Stats('output.prof')
# p.sort_stats('cumulative').print_stats(20)