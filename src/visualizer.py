import matplotlib.pyplot as plt
import os
import numpy as np

class Visualizer:
    def __init__(self, coords, best_distance, best_route, execution_time, dataset_name="Unknown", algorithm="Unknown"):
        """
        Initializes and immediately plots the TSP route.
        """
        if coords is None or len(coords) == 0:
            print(f"  - [Visualizer] Skipping plot for {dataset_name}: No coordinate data available.")
            return

        self.coords = np.array(coords)
        self.best_distance = best_distance
        self.best_route = best_route
        self.execution_time = execution_time
        self.dataset_name = dataset_name
        self.algorithm = algorithm
        
        self._plot_and_save()

    def _plot_and_save(self):
        # Create a high-resolution figure suitable for publication
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Extract X and Y coordinates in the order of the route
        route_coords = self.coords[self.best_route]
        
        # Append the starting city to the end to close the TSP loop
        route_coords = np.vstack((route_coords, self.coords[self.best_route[0]]))
        
        x = route_coords[:, 0]
        y = route_coords[:, 1]
        
        # Plot the paths (edges)
        plt.plot(x, y, linestyle='-', color='royalblue', linewidth=1.5, alpha=0.8, zorder=1, label='Tour Path')
        
        # Plot the cities (nodes)
        plt.scatter(self.coords[:, 0], self.coords[:, 1], color='darkorange', s=25, edgecolor='black', zorder=2, label='Cities')
        
        # Highlight the starting/ending node with a distinct marker
        plt.scatter(x[0], y[0], color='crimson', s=150, marker='*', edgecolor='black', zorder=3, label='Start/End Node')
        
        # Formatting for readability
        plt.title(f'TSP Tour: {self.dataset_name} ({self.algorithm.upper()})\nDistance: {self.best_distance:.2f} | Time: {self.execution_time:.4f}s', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('X Coordinate', fontsize=11)
        plt.ylabel('Y Coordinate', fontsize=11)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Remove extra whitespace
        plt.tight_layout()
        
        # Save the figure dynamically based on dataset and algorithm
        output_dir = os.path.join('results', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.dataset_name}_{self.algorithm}_tour.png"
        
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        print(f"  - Route visualization saved to {output_dir}/{filename}")
        
        # Close the plot to free memory, preventing slowdowns during large batch runs
        plt.close()