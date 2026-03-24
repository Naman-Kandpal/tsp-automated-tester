import numpy as np

def solve(distance_matrix, **kwargs):
    """
    Solves the TSP using the Nearest Neighbour heuristic.
    It evaluates routes starting from every possible city and returns the best overall tour.
    
    Args:
        distance_matrix: 2D array or list of lists representing distances between cities.
        **kwargs: Absorbs any extra arguments (like pop_size) passed by main.py so it doesn't crash.
        
    Returns:
        tuple: (best_route, best_distance)
    """
    num_cities = len(distance_matrix)
    
    # Fast O(1) native Python list indexing, just like we did for the GA
    matrix_list = distance_matrix.tolist() if isinstance(distance_matrix, np.ndarray) else distance_matrix

    best_overall_distance = float('inf')
    best_overall_route = []

    # Try starting from every single city to find the best possible NN route
    for start_city in range(num_cities):
        unvisited = set(range(num_cities))
        unvisited.remove(start_city)
        
        route = [start_city]
        current_city = start_city
        current_distance = 0

        # Build the route by always jumping to the closest unvisited city
        while unvisited:
            nearest_city = None
            min_dist = float('inf')
            
            # Explicit loop is faster here than min(key=lambda) due to avoiding function call overhead
            for city in unvisited:
                dist = matrix_list[current_city][city]
                if dist < min_dist:
                    min_dist = dist
                    nearest_city = city
            
            # Move to the nearest city
            current_distance += min_dist
            route.append(nearest_city)
            current_city = nearest_city
            unvisited.remove(current_city)

        # Complete the tour by returning to the start city
        current_distance += matrix_list[current_city][start_city]

        # Keep track of the best tour found across all starting points
        if current_distance < best_overall_distance:
            best_overall_distance = current_distance
            best_overall_route = route

    return best_overall_route, best_overall_distance