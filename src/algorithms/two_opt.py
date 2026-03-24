import numpy as np
from src.algorithms.nearest_neighbour import solve as solve_nn

def calculate_route_distance(route, matrix_list):
    """Calculates total distance using fast native Python list indexing."""
    total_dist = 0
    num_cities = len(route)
    for i in range(num_cities):
        total_dist += matrix_list[route[i]][route[(i + 1) % num_cities]]
    return total_dist

def solve(distance_matrix, initial_route=None, **kwargs):
    """
    Solves the TSP using the 2-Opt Local Search algorithm.
    It takes an initial route (defaulting to Nearest Neighbour) and untangles crossing edges.
    """
    num_cities = len(distance_matrix)
    
    # Fast native Python list indexing for speed
    matrix_list = distance_matrix.tolist() if isinstance(distance_matrix, np.ndarray) else distance_matrix

    # If no initial route is provided, use Nearest Neighbour to get a strong starting point
    if initial_route is None:
        best_route, _ = solve_nn(distance_matrix)
    else:
        best_route = initial_route[:]

    best_distance = calculate_route_distance(best_route, matrix_list)
    improvement = True

    # Continue untangling as long as we find shorter paths
    while improvement:
        improvement = False
        
        for i in range(1, num_cities - 1):
            for k in range(i + 1, num_cities):
                # Identify the four cities involved in the two edges being considered for a swap
                c1 = best_route[i - 1]
                c2 = best_route[i]
                c3 = best_route[k]
                c4 = best_route[(k + 1) % num_cities]

                # Calculate the distance of the current edges vs the proposed new edges
                current_edges_weight = matrix_list[c1][c2] + matrix_list[c3][c4]
                new_edges_weight = matrix_list[c1][c3] + matrix_list[c2][c4]

                # SPEED HACK: Only perform the slice/reverse operation if it mathematically improves the route
                if new_edges_weight < current_edges_weight:
                    # Perform the 2-Opt swap: reverse the segment between i and k
                    best_route[i:k+1] = reversed(best_route[i:k+1])
                    
                    # Update the total distance without recalculating the whole route
                    best_distance = best_distance - current_edges_weight + new_edges_weight
                    improvement = True
                    
                    # Break out to restart the search from the beginning (First Improvement strategy)
                    break 
            
            if improvement:
                break

    return best_route, best_distance