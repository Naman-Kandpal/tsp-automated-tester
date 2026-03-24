import os
import numpy as np
import tsplib95

def get_problem(dataset_name):
    """
    Loads a TSP problem using tsplib95.
    Robustly handles all explicit and implicit matrix formats.
    """
    possible_extensions = ['.tsp', '.atsp', '.hcp']
    problem_path = None
    
    # Search in coordinates directory
    for ext in possible_extensions:
        path = os.path.join('data', 'coordinates', f'{dataset_name}{ext}')
        if os.path.exists(path):
            problem_path = path
            break

    # Fallback to distance_matrices directory
    if not problem_path:
        dist_matrix_path = os.path.join('data', 'distance_matrices', f'{dataset_name}.txt')
        if os.path.exists(dist_matrix_path):
            problem_path = dist_matrix_path
        else:
            raise FileNotFoundError(f"Could not find data for '{dataset_name}'.")

    # If it's a raw custom text matrix, bypass tsplib95 entirely
    if problem_path.endswith('.txt'):
        dist_matrix = load_simple_matrix(problem_path)
        return None, dist_matrix

    print(f"Loading problem from {problem_path}")
    problem = tsplib95.load(problem_path)

    # Extract coordinates for the Visualizer if they exist
    coords = None
    if problem.is_depictable():
        node_coords = problem.node_coords or problem.display_data
        if node_coords:
            coords = np.array([list(node_coords[i]) for i in sorted(node_coords.keys())])

    # --- UNIVERSAL MATRIX GENERATOR ---
    # We iterate through the nodes and let tsplib95 handle the underlying format logic.
    nodes = list(problem.get_nodes())
    n = len(nodes)
    dist_matrix = np.zeros((n, n), dtype=int)
    
    try:
        for i in range(n):
            for j in range(n):
                # get_weight automatically resolves UPPER_ROW, EUC_2D, GEO, FULL_MATRIX, etc.
                dist_matrix[i, j] = problem.get_weight(nodes[i], nodes[j])
    except Exception as e:
        raise ValueError(f"Failed to extract distances from '{dataset_name}': {e}")

    return coords, dist_matrix

def load_simple_matrix(file_path):
    # ... [Keep your existing load_simple_matrix code exactly as it is] ...
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
        
    n_str = lines[0]
    if len(n_str.split()) == 1:
        n = int(n_str)
        data_lines = [list(map(int, line.split())) for line in lines[1:]]
    else: 
        data_lines = [list(map(int, line.split())) for line in lines]
        n = len(data_lines)

    is_triangular = all(len(data_lines[i]) == i + 1 for i in range(len(data_lines)))

    if is_triangular:
        matrix = np.zeros((n, n), dtype=int)
        for i, row in enumerate(data_lines):
            for j, val in enumerate(row):
                matrix[i, j] = matrix[j, i] = val
    else:
        matrix = np.array(data_lines, dtype=int)
            
    return matrix

def load_optimal_solutions(file_path=os.path.join('data', 'coordinates', 'solutions')):
    # ... [Keep your existing load_optimal_solutions code exactly as it is] ...
    solutions = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ' : ' in line:
                name, value = line.split(' : ')
                value = value.split(' ')[0]
                solutions[name.strip()] = int(value.strip())
    return solutions