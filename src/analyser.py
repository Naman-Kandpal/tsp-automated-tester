import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

# Define file paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
SUMMARY_DIR = os.path.join(RESULTS_DIR, 'summaries')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Set publication-quality plot theme
sns.set_theme(style="whitegrid", palette="muted")

def load_results(filename):
    """Loads and sanitizes the results csv file."""
    results_csv_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(results_csv_path):
        print(f"Error: Results file not found at {results_csv_path}")
        return None
        
    df = pd.read_csv(results_csv_path)
    
    # Fix string formatting from main.py and convert to numeric
    for col in ['best_distance', 'optimal_distance', 'accuracy', 'time_seconds']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Calculate Optimality Gap (Percentage Error)
    # We use np.where to prevent division by zero if optimal_distance is missing/0
    df['optimality_gap_pct'] = np.where(
        df['optimal_distance'] > 0,
        ((df['best_distance'] - df['optimal_distance']) / df['optimal_distance']) * 100,
        np.nan
    )
    
    # Flag if the algorithm found the exact optimal solution (Gap < 0.01%)
    df['optimal_hit'] = (df['optimality_gap_pct'] < 0.01).astype(int)
    
    return df

def generate_summary_statistics(df):
    """Generates deep analytical summaries and exports them to CSV."""
    if df is None or df.empty:
        return None

    print("\n" + "="*50)
    print("GLOBAL ALGORITHM PERFORMANCE SUMMARY")
    print("="*50)
    
    # 1. Global Macro Summary (Aggregated across all datasets)
    global_summary = df.groupby('algorithm').agg(
        total_runs=('run', 'count'),
        avg_gap_pct=('optimality_gap_pct', 'mean'),
        median_gap_pct=('optimality_gap_pct', 'median'),
        avg_time_s=('time_seconds', 'mean'),
        optimal_hits=('optimal_hit', 'sum')
    ).reset_index()
    
    # Calculate hit rate percentage
    global_summary['hit_rate_pct'] = (global_summary['optimal_hits'] / global_summary['total_runs']) * 100
    
    # Format for printing
    print(global_summary.to_string(index=False, float_format="%.2f"))
    global_summary.to_csv(os.path.join(SUMMARY_DIR, 'global_summary.csv'), index=False)
    print(f"\n[Saved Global Summary: {os.path.join(SUMMARY_DIR, 'global_summary.csv')}]")

    print("\n" + "="*50)
    print("DETAILED PER-DATASET SUMMARY")
    print("="*50)
    
    # 2. Detailed Micro Summary (Per Dataset, Per Algorithm)
    detailed_summary = df.groupby(['dataset', 'algorithm']).agg(
        runs=('run', 'count'),
        mean_gap=('optimality_gap_pct', 'mean'),
        best_gap=('optimality_gap_pct', 'min'),
        worst_gap=('optimality_gap_pct', 'max'),
        mean_time=('time_seconds', 'mean'),
        hits=('optimal_hit', 'sum')
    ).reset_index()

    print(detailed_summary.head(15).to_string(index=False, float_format="%.2f"))
    if len(detailed_summary) > 15:
        print(f"... and {len(detailed_summary) - 15} more rows.")
        
    detailed_summary.to_csv(os.path.join(SUMMARY_DIR, 'detailed_summary.csv'), index=False)
    print(f"\n[Saved Detailed Summary: {os.path.join(SUMMARY_DIR, 'detailed_summary.csv')}]")

    return global_summary, detailed_summary

def generate_plots(df, detailed_summary):
    """Generates comparative analytical plots."""
    if df is None or detailed_summary is None:
        return

    print("\nGenerating comparative plots...")

    # PLOT 1: Global Optimality Gap Distribution (Violin + Boxplot combo)
    plt.figure(figsize=(10, 6), dpi=300)
    sns.violinplot(x='algorithm', y='optimality_gap_pct', data=df, inner="quartile", alpha=0.6)
    sns.stripplot(x='algorithm', y='optimality_gap_pct', data=df, size=3, color="black", alpha=0.3)
    plt.title('Algorithm Error Distribution (All Datasets)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Optimality Gap (%)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'global_error_distribution.png'))
    plt.close()

    # PLOT 2: Time vs Performance Trade-off (Scatter Pareto Front)
    plt.figure(figsize=(10, 6), dpi=300)
    sns.scatterplot(
        data=detailed_summary, x='mean_time', y='mean_gap', 
        hue='algorithm', style='algorithm', s=100, alpha=0.8
    )
    plt.title('Time vs. Optimality Trade-off (Lower Left is Better)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Average Optimality Gap (%)', fontsize=12)
    plt.xlabel('Average Execution Time (Seconds)', fontsize=12)
    # Log scale is usually better for time as GA will be much slower than NN/2Opt
    plt.xscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_vs_performance_tradeoff.png'))
    plt.close()

    # PLOT 3: Per-Dataset Bar Charts for direct comparison
    datasets = df['dataset'].unique()
    for dataset in datasets:
        plt.figure(figsize=(8, 5), dpi=300)
        dataset_df = df[df['dataset'] == dataset]
        
        sns.barplot(x='algorithm', y='optimality_gap_pct', data=dataset_df, capsize=.1, errorbar='sd')
        plt.title(f'Performance Comparison: {dataset.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('Optimality Gap (%) [Lower is Better]', fontsize=11)
        plt.xlabel('Algorithm', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{dataset}_comparison.png'))
        plt.close()

    print(f"All plots successfully saved to {PLOTS_DIR}/")

def main():
    parser = argparse.ArgumentParser(description="Analyze TSP experiment results.")
    parser.add_argument('--file', type=str, default='all_runs.csv',
                        help='The name of the results CSV file to analyze (default: all_runs.csv)')
    args = parser.parse_args()

    results_df = load_results(args.file)
    if results_df is not None:
        global_summary, detailed_summary = generate_summary_statistics(results_df)
        generate_plots(results_df, detailed_summary)

if __name__ == '__main__':
    main()