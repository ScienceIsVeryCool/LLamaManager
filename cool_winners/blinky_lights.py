import pygame
import random
import math
from collections import deque
from dataclasses import dataclass
import cProfile
import pstats

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 50
FOOD_ENERGY = 10
MUTATION_RATE = 0.05
INITIAL_POPULATION = 50
FPS = 30
POPULATION_CARRYING_CAPACITY = 200
DEATH_RATE_MODIFIER = 0.0001
FOOD_RESPAWN_INTERVAL = 5  # Seconds between food respawns

@dataclass
class Size:
    initial: float = 1.0
    mutation_range: float = 0.1

@dataclass
class Speed:
    initial: float = 1.0
    mutation_range: float = 0.1

@dataclass
class EnergyConsumptionRate:
    initial: float = 0.1
    mutation_range: float = 0.02


class Organism:
    def __init__(self, x, y, genome):
        """
        Initializes an organism.

        Args:
            x (int): X-coordinate of the organism.
            y (int): Y-coordinate of the organism.
            genome (dict): Dictionary representing the organism's genome.
        """
        self.x = x
        self.y = y
        self.genome = genome
        self.size = self.genome['size'].initial
        self.speed = self.genome['speed'].initial
        self.energy_consumption_rate = self.genome['energy_consumption_rate'].initial
        self.energy = 100  # Initial energy
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update(self, food_sources):
        """
        Updates the organism's state.

        Args:
            food_sources (list): List of Food objects.

        Returns:
            bool: True if the organism is alive, False otherwise.
        """
        # Energy consumption
        self.energy -= self.energy_consumption_rate

        if self.energy <= 0:
            return False  # Organism dies

        # Find closest food source (optimization)
        closest_food = None
        min_distance = float('inf')
        for food in food_sources:
            distance = math.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_food = food

        if closest_food:
            # Move towards the closest food (simplified movement)
            dx = closest_food.x - self.x
            dy = closest_food.y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 0:
                self.x += dx * self.speed / distance
                self.y += dy * self.speed / distance

            # Eat food if close enough
            if math.sqrt((self.x - closest_food.x)**2 + (self.y - closest_food.y)**2) < self.size:
                self.energy += 150  # Replenish energy
                closest_food.respawn(SCREEN_WIDTH, SCREEN_HEIGHT) # Respawn the food

        return True

    def reproduce(self):
        """Reproduces a new organism with slight mutations."""
        new_size = self.genome['size'].initial + random.uniform(-self.genome['size'].mutation_range, self.genome['size'].mutation_range)
        new_speed = self.genome['speed'].initial + random.uniform(-self.genome['speed'].mutation_range, self.genome['speed'].mutation_range)
        new_energy_consumption_rate = self.genome['energy_consumption_rate'].initial + random.uniform(-self.genome['energy_consumption_rate'].mutation_range, self.genome['energy_consumption_rate'].mutation_range)

        return Organism(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), {
            'size': Size(initial=new_size, mutation_range=self.genome['size'].mutation_range),
            'speed': Speed(initial=new_speed, mutation_range=self.genome['speed'].mutation_range),
            'energy_consumption_rate': EnergyConsumptionRate(initial=new_energy_consumption_rate, mutation_range=self.genome['energy_consumption_rate'].mutation_range)
        })


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def respawn(self, screen_width, screen_height):
        self.x = random.randint(0, screen_width)
        self.y = random.randint(0, screen_height)


class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Evolution Simulation")
        self.clock = pygame.time.Clock()
        self.food_respawn_time = 0  # Time since last food respawn
        self.food_respawn_interval = FOOD_RESPAWN_INTERVAL
        self.population_history = []
        self.average_size_history = []

    def run(self):
        self.initialize()
        running = True
        while running:
            start_time = pygame.time.get_ticks()  # Start time for update method
            self.handle_events()
            self.update()
            end_time = pygame.time.get_ticks()  # End time for update method
            update_time = (end_time - start_time) / 1000.0  # Update time in seconds
            #print(f"Update time: {update_time:.4f} seconds")  # Commented out print statement
            self.clock.tick(FPS)
            self.render()

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return

    def update(self):
        # Organism updates
        dead_organisms = []
        for organism in self.organisms:
            if not organism.update(self.food_sources):
                dead_organisms.append(organism)

        # Remove dead organisms
        for organism in dead_organisms:
            self.organisms.remove(organism)

        # Reproduction
        new_organisms = []
        for organism in self.organisms:
            if organism.energy > 150:  # Increased energy threshold for reproduction
                offspring = organism.reproduce()
                new_organisms.append(offspring)
                organism.energy = 150  # Reset energy after reproduction

        self.organisms.extend(new_organisms)

        # Food respawn - using a timer for more controlled respawning
        self.food_respawn_time += 1  # Increment timer
        if self.food_respawn_time >= self.food_respawn_interval:
            self.respawn_food()
            self.food_respawn_time = 0  # Reset timer

        # Population control - simple death rate based on population size
        population_size = len(self.organisms)
        death_rate = 0.01 + DEATH_RATE_MODIFIER * (population_size - POPULATION_CARRYING_CAPACITY)
        if population_size > POPULATION_CARRYING_CAPACITY:
            for i in range(len(self.organisms)):
                if random.random() < death_rate:
                    self.organisms.pop(i)
                    break

        # Record population history
        self.population_history.append(len(self.organisms))
        if self.organisms:
            self.average_size = sum([organism.size for organism in self.organisms]) / len(self.organisms)
        else:
            self.average_size = 0
        self.average_size_history.append(self.average_size)

    def respawn_food(self):
        """Respawn all food items randomly on the screen."""
        for food in self.food_sources:
            food.respawn(SCREEN_WIDTH, SCREEN_HEIGHT)

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw organisms
        for organism in self.organisms:
            energy_ratio = organism.energy / 200
            color = (int(255 * energy_ratio), int(255 * (1 - energy_ratio)), 0)
            pygame.draw.circle(self.screen, color, (int(organism.x), int(organism.y)), int(organism.size))

        # Draw food
        for food in self.food_sources:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(food.x), int(food.y)), 2)

        # Display information
        font = pygame.font.Font(None, 20)
        population_text = font.render(f"Population: {len(self.organisms)}", True, (255, 255, 255))
        average_size_text = font.render(f"Average Size: {self.average_size}", True, (255, 255, 255))
        self.screen.blit(population_text, (10, 10))
        self.screen.blit(average_size_text, (10, 30))

        pygame.display.flip()

    def initialize(self):
        """Initializes the simulation."""
        self.organisms = []
        self.food_sources = []
        self.population_history = []
        self.average_size_history = []

        # Initial population
        for _ in range(INITIAL_POPULATION):
            genome = {
                'size': Size(),
                'speed': Speed(),
                'energy_consumption_rate': EnergyConsumptionRate()
            }
            self.organisms.append(Organism(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), genome))

        # Initial food sources
        for _ in range(50):
            self.food_sources.append(Food(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)))


if __name__ == "__main__":
    #simulation = Simulation()
    #cProfile.run('simulation.run()') #Profile the simulation
    simulation = Simulation()
    simulation.run()