import random

# --- Core Solver ---
def solve(distance_matrix, pop_size, tournament_size, crossover_prob, inversion_prob, exchange_prob, num_generations, elitism_ratio):
    num_cities = len(distance_matrix)
    
    # Fast O(1) native Python list indexing
    matrix_list = distance_matrix.tolist()

    population = initialize_population(pop_size, num_cities)
    elitism_count = int(elitism_ratio * pop_size)
    offspring_count = pop_size - elitism_count 

    for _ in range(num_generations):
        # Generate offspring, ensuring the exact target size to prevent shrinking
        new_population = _generate_offspring(population, matrix_list, tournament_size, crossover_prob, inversion_prob, exchange_prob, len(population))
        
        new_population.sort(key=lambda x: total_distance(x, matrix_list))
        elite_individuals = new_population[:elitism_count]
        
        non_elite_population = new_population[elitism_count:]
        offspring_population = _generate_offspring(non_elite_population, matrix_list, tournament_size, crossover_prob, inversion_prob, exchange_prob, offspring_count)
        
        population = elite_individuals + offspring_population

    best_route = min(population, key=lambda x: total_distance(x, matrix_list))
    best_distance = total_distance(best_route, matrix_list)

    return best_route, best_distance

# --- Main Helper for Generation ---
def _generate_offspring(population, matrix_list, tournament_size, crossover_prob, inversion_prob, exchange_prob, target_size):
    new_population = []
    
    while len(new_population) < target_size:
        parent1 = tournament_selection(population, matrix_list, tournament_size)
        parent2 = tournament_selection(population, matrix_list, tournament_size)

        if random.random() < crossover_prob:
            child1 = pmx_crossover(parent1, parent2)
            child2 = pmx_crossover(parent2, parent1)
        else:
            child1, child2 = parent1[:], parent2[:]

        if random.random() < inversion_prob: child1 = inversion_mutation(child1)
        if random.random() < inversion_prob: child2 = inversion_mutation(child2)
        if random.random() < exchange_prob: child1 = exchange_mutation(child1)
        if random.random() < exchange_prob: child2 = exchange_mutation(child2)

        if total_distance(child1, matrix_list) < total_distance(child2, matrix_list):
            new_population.append(child1)
            if len(new_population) < target_size:
                new_population.append(parent2)
        else:
            new_population.append(child2)
            if len(new_population) < target_size:
                new_population.append(parent1)
                
    return new_population

# --- Core GA Components ---
def total_distance(route, matrix_list):
    total_dist = 0
    num_cities = len(route)
    for i in range(num_cities):
        total_dist += matrix_list[route[i]][route[(i + 1) % num_cities]]
    return total_dist

def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

def tournament_selection(population, matrix_list, k):
    selected = random.sample(population, k)
    return min(selected, key=lambda x: total_distance(x, matrix_list))

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    child = parent1[a:b+1]
    child_set = set(child)

    # FIX: O(1) mapping entirely eliminates the .index() exponential slowdown
    next_gene_map = {parent2[i]: parent2[(i + 1) % size] for i in range(size)}

    for i in range(size):
        if i < a or i > b:
            gene = parent2[i]
            while gene in child_set:
                gene = next_gene_map[gene] # Instant O(1) lookup
            child.append(gene)
            child_set.add(gene)

    return child

def inversion_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    if a > b: a, b = b, a
    route[a:b+1] = reversed(route[a:b+1])
    return route

def exchange_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route