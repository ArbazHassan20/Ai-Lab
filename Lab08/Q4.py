import numpy as np
from itertools import product

states = ["Sunny", "Cloudy", "Rainy"]

transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.4, 0.4, 0.2],
    [0.2, 0.6, 0.2]
])

def simulate_markov_process(initial_state_index, num_steps, transition_matrix, states):
    current_state_index = initial_state_index
    state_sequence = [states[current_state_index]]
    state_indices = [current_state_index]
    
    for _ in range(num_steps):
        next_state_index = np.random.choice(
            len(states), 
            p=transition_matrix[current_state_index]
        )
        state_sequence.append(states[next_state_index])
        state_indices.append(next_state_index)
        current_state_index = next_state_index
    
    return state_sequence, state_indices

initial_state_index = 0
num_steps = 9
num_simulations = 10000

state_sequence, _ = simulate_markov_process(
    initial_state_index, 
    num_steps, 
    transition_matrix, 
    states
)

print(f"Example weather sequence for 10 days starting from {states[initial_state_index]}:")
print(" -> ".join(state_sequence))
print()

count_at_least_3_rainy = 0

for _ in range(num_simulations):
    _, state_indices = simulate_markov_process(
        initial_state_index, 
        num_steps, 
        transition_matrix, 
        states
    )
    rainy_count = state_indices.count(2)
    if rainy_count >= 3:
        count_at_least_3_rainy += 1

prob_at_least_3_rainy = count_at_least_3_rainy / num_simulations
print(f"Probability of at least 3 rainy days in 10 days: {prob_at_least_3_rainy:.4f}")
