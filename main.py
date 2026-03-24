import argparse
import time
import os
import pandas as pd
import glob
from src.algorithms.genetic_algorithm import solve as solve_ga
from src.algorithms.nearest_neighbour import solve as solve_nn
from src.algorithms.two_opt import solve as solve_2opt

from src.data_parser import get_problem, load_optimal_solutions
from src.visualizer import Visualizer

def get_available_datasets(data_dir):
    """Scans the data directory for available .tsp files."""
    tsp_files = glob.glob(os.path.join(data_dir, '*.tsp'))
    return [os.path.basename(f).replace('.tsp', '') for f in tsp_files]

def main():
    parser = argparse.ArgumentParser(description='Run TSP solver experiments.')
    parser.add_argument('--dataset', type=str, help='Specify a single dataset to run (e.g., berlin52).')
    parser.add_argument('--min-cities', type=int, default=0, help='Minimum number of cities for datasets to be included in the run.')
    parser.add_argument('--max-cities', type=int, default=150, help='Maximum number of cities for datasets to be included in the run.')
    parser.add_argument('--algorithms', nargs='+', required=True, choices=['ga', 'nn', '2opt'], help='List of algorithms to run (e.g., ga nn 2opt).')
    parser.add_argument('--runs', type=int, default=30, help='Number of runs per algorithm per dataset.')
    parser.add_argument('--output-file', type=str, default='results/all_runs.csv', help='File to append all results to.')
    parser.add_argument('--visualize', action='store_true', help='Visualize the best tour of the last run for each dataset.')
    args = parser.parse_args()

    # Load optimal solutions
    optimal_solutions = load_optimal_solutions()

    # Determine which datasets to run
    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        all_datasets = get_available_datasets(os.path.join('data', 'coordinates'))
        datasets_to_run = []
        print("Filtering datasets...")
        for ds in all_datasets:
            try:
                _, dist_matrix = get_problem(ds)
                num_cities = len(dist_matrix)
                if args.min_cities <= num_cities <= args.max_cities:
                    datasets_to_run.append(ds)
                    print(f"  - Included: {ds} ({num_cities} cities)")
            except Exception as e:
                print(f"  - Could not load {ds}. Error: {e}")
        print(f"\nSelected {len(datasets_to_run)} datasets to run.\n")


    # Prepare results file
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    
    if not os.path.exists(args.output_file):
        # Create header for the master CSV
        header = pd.DataFrame(columns=['dataset', 'algorithm', 'run', 'best_distance', 'optimal_distance', 'accuracy', 'time_seconds'])
        header.to_csv(args.output_file, index=False)

    # Main experiment loop
    for dataset in datasets_to_run:
        print(f"--- Running Dataset: {dataset} ---")
        try:
            coords, dist_matrix = get_problem(dataset)
            optimal_dist = optimal_solutions.get(dataset, 0)
        except Exception as e:
            print(f"Could not process {dataset}. Skipping. Error: {e}")
            continue

        for algorithm in args.algorithms:
            print(f"  - Algorithm: {algorithm}")
            
            all_run_results = []
            for run in range(1, args.runs + 1):
                start_time = time.time()
                
                if algorithm == 'ga':
                    # GA parameters
                    pop_size, num_generations = (100, 1000) if len(dist_matrix) < 100 else (200, 2000)
                    best_route, best_distance = solve_ga(
                        dist_matrix, pop_size=pop_size, tournament_size=3, crossover_prob=0.85,
                        inversion_prob=0.15, exchange_prob=0.15, num_generations=num_generations, elitism_ratio=0.05
                    )
                elif algorithm == 'nn':
                    best_route, best_distance = solve_nn(dist_matrix)
                elif algorithm == '2opt':
                    best_route, best_distance = solve_2opt(dist_matrix)
                else:
                    print(f"Algorithm '{algorithm}' not implemented. Skipping.")
                    continue
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                accuracy = (optimal_dist / best_distance) * 100 if best_distance > 0 and optimal_dist > 0 else 0

                print(f"    Run {run}/{args.runs} | Best: {best_distance:.2f} | Optimal: {optimal_dist} | Acc: {accuracy:.2f}% | Time: {execution_time:.2f}s")
                
                all_run_results.append([dataset, algorithm, run, best_distance, optimal_dist, f"{accuracy:.2f}", f"{execution_time:.4f}"])

            # Append all results for this algorithm-dataset pair to the CSV
            df = pd.DataFrame(all_run_results, columns=['dataset', 'algorithm', 'run', 'best_distance', 'optimal_distance', 'accuracy', 'time_seconds'])
            df.to_csv(args.output_file, mode='a', header=False, index=False)
            print(f"  - Results for {algorithm} on {dataset} saved to {args.output_file}")

            if args.visualize:
                print("  - Visualizing best tour from the last run...")
                Visualizer(coords, best_distance, best_route, execution_time, dataset_name=dataset, algorithm=algorithm)

if __name__ == '__main__':
    main()
