import random
import math

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(route):
    """Calculate total distance of the given route."""
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1)) + distance(route[-1], route[0])

def hill_climb(delivery_points, max_iterations=1000):
    """Hill climbing algorithm to find the shortest delivery route."""
    current_route = delivery_points[:]
    random.shuffle(current_route)  # Start with a random route
    current_distance = total_distance(current_route)
    
    for _ in range(max_iterations):
        new_route = current_route[:]
        i, j = random.sample(range(len(new_route)), 2) 
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_distance = total_distance(new_route)
        
        if new_distance < current_distance:
            current_route, current_distance = new_route, new_distance  
    
    return current_route, current_distance

# Example usage
delivery_points = [(0, 0), (2, 3), (5, 2), (7, 8), (1, 6)]  
best_route, best_distance = hill_climb(delivery_points)
print("Optimized Route:", best_route)
print("Total Distance:", best_distance)
