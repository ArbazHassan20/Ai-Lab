import random
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(route):
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1)) + distance(route[-1], route[0])

def create_population(cities, size=100):
    return [random.sample(cities, len(cities)) for _ in range(size)]

def selection(population):
    return sorted(random.sample(population, 5), key=total_distance)[:2]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]
    child = [remaining.pop(0) if city is None else city for city in child]
    return child

def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(cities, population_size=100, generations=500):
    population = create_population(cities, population_size)
    
    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        population = sorted(new_population, key=total_distance)[:population_size]
    
    best_route = min(population, key=total_distance)
    return best_route, total_distance(best_route)

cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
best_route, best_distance = genetic_algorithm(cities)
print("Optimized Route:", best_route)
print("Total Distance:", best_distance)
