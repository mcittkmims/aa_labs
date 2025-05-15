import os
import time
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque

# Import the helpers
from helpers.bfs import bfs, measure_bfs_performance
from helpers.dfs import dfs, measure_dfs_performance
from helpers.dijkstra import dijkstra, measure_dijkstra_performance, get_shortest_path
from helpers.floyd_warshall import floyd_warshall, measure_floyd_warshall_performance, get_path
from helpers.kruskal import kruskal, measure_kruskal_performance, test_kruskal
from helpers.prim import prim, measure_prim_performance, test_prim, find_largest_component
from helpers.graph_builders import (
    generate_complete_graph,
    generate_dense_graph,
    generate_sparse_graph,
    generate_tree_graph,
    generate_connected_graph,
    generate_disconnected_graph,
    generate_cyclic_graph,
    generate_acyclic_graph,
    generate_grid_graph
)

# Increase recursion limit for large graphs
sys.setrecursionlimit(16000)

# Get the directory where the script file is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Create output directories relative to the script location
output_folders = ["graphs/lab3", "graphs/lab4", "graphs/lab5"]
for folder in output_folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# Graph types with their generator functions
GRAPH_VARIANTS = {
    "Complete Graph": generate_complete_graph,
    "Dense Graph": generate_dense_graph,
    "Sparse Graph": generate_sparse_graph,
    "Tree Graph": generate_tree_graph,
    "Connected Graph": generate_connected_graph,
    "Disconnected Graph": generate_disconnected_graph,
    "Cyclic Graph": generate_cyclic_graph,
    "Acyclic Graph": generate_acyclic_graph,
    "Grid Graph": generate_grid_graph
}

# Color schemes for different algorithms
COLOR_SCHEMES = {
    "bfs": "darkmagenta",
    "dfs": "navy", 
    "dijkstra": "darkorchid",
    "floyd_warshall": "slateblue",
    "kruskal": "royalblue",
    "prim": "forestgreen"
}

# Common utility functions
def add_weights(adj_list, min_weight=1, max_weight=10):
    """Convert an unweighted adjacency list to a weighted one"""
    weighted_adj = [{} for _ in range(len(adj_list))]
    for u in range(len(adj_list)):
        for v in adj_list[u]:
            if v not in weighted_adj[u]:  # Avoid duplicating edges
                weight = random.randint(min_weight, max_weight)
                weighted_adj[u][v] = weight
                weighted_adj[v][u] = weight  # Ensure graph is undirected for MST
    return weighted_adj

def make_undirected(directed_adj):
    """Make a directed graph undirected for MST algorithms"""
    n = len(directed_adj)
    undirected = [[] for _ in range(n)]
    for u in range(n):
        for v in directed_adj[u]:
            undirected[u].append(v)
            undirected[v].append(u)  # Add reverse edge
    
    # Remove duplicates
    for i in range(n):
        undirected[i] = list(set(undirected[i]))
    return undirected

def format_time(time_ms):
    """Format time nicely with appropriate units"""
    if time_ms < 0.1:
        return f"{time_ms*1000:.2f}μs"
    elif time_ms < 1000:
        return f"{time_ms:.2f}ms"
    else:
        return f"{time_ms/1000:.2f}s"

class GraphAnalyzer:
    """Main class handling the unified lab functionality"""
    
    def __init__(self):
        # Dictionary to store results for each algorithm type
        self.performance_data = {
            "bfs": {},
            "dfs": {},
            "dijkstra": {},
            "floyd_warshall": {},
            "kruskal": {},
            "prim": {}
        }
        # Store script directory for saving results
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        
    def run_traversal_tests(self, sizes):
        """Run BFS and DFS tests (Lab 3)"""
        print("\n--- Running Graph Traversal Tests (BFS/DFS) ---")
        
        results_bfs = {graph_type: [] for graph_type in GRAPH_VARIANTS}
        results_dfs = {graph_type: [] for graph_type in GRAPH_VARIANTS}

        for size in sizes:
            print(f"Testing graphs with {size} nodes...")
            for graph_type, generator_func in GRAPH_VARIANTS.items():
                print(f"  {graph_type}...")

                # Generate graph
                graph = generator_func(size)

                # Test BFS
                bfs_metrics = measure_bfs_performance(graph, start_node=0)
                bfs_time = bfs_metrics["execution_time"]

                # Test DFS
                dfs_metrics = measure_dfs_performance(graph, start_node=0)
                dfs_time = dfs_metrics["execution_time"]

                # Store results
                results_bfs[graph_type].append(bfs_time)
                results_dfs[graph_type].append(dfs_time)
                print(f"    BFS={format_time(bfs_time)}, DFS={format_time(dfs_time)}")

        # Store the results
        self.performance_data["bfs"] = results_bfs
        self.performance_data["dfs"] = results_dfs
        
        # Plot the results
        self.plot_traversal_results(sizes)
        return results_bfs, results_dfs
    
    def plot_traversal_results(self, node_counts):
        """Generate plots for BFS and DFS results"""
        bfs_data = self.performance_data["bfs"]
        dfs_data = self.performance_data["dfs"]
        
        # Create a figure with two vertically stacked subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: BFS performance (top)
        for graph_type, times in bfs_data.items():
            axes[0].plot(node_counts, times, marker='o', linewidth=2, 
                      label=graph_type, alpha=0.8)

        axes[0].set_title('BFS Performance Across Graph Types', fontsize=14, weight='bold')
        axes[0].set_xlabel('Number of Nodes', fontsize=12)
        axes[0].set_ylabel('Time (ms)', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper left', framealpha=0.9)

        # Plot 2: DFS performance (bottom)
        for graph_type, times in dfs_data.items():
            axes[1].plot(node_counts, times, marker='*', linewidth=2, 
                      label=graph_type, alpha=0.8)

        axes[1].set_title('DFS Performance Across Graph Types', fontsize=14, weight='bold')
        axes[1].set_xlabel('Number of Nodes', fontsize=12)
        axes[1].set_ylabel('Time (ms)', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper left', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "graphs/lab3/traversal_algorithm_comparison.png"), dpi=300)
        plt.close()

        # Individual plots for each graph type
        for graph_type in GRAPH_VARIANTS:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(node_counts, bfs_data[graph_type], marker='o', linewidth=2.5,
                 label='BFS', color=COLOR_SCHEMES['bfs'], alpha=0.8)
            ax.plot(node_counts, dfs_data[graph_type], marker='*', linewidth=2.5,
                 label='DFS', color=COLOR_SCHEMES['dfs'], alpha=0.8)

            ax.set_title(f'Traversal Algorithms on {graph_type}', fontsize=14, weight='bold')
            ax.set_xlabel('Number of Nodes', fontsize=12)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11)
            
            # Add a text box with the graph type info
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
            info_text = f"Graph Type: {graph_type}\nNodes: {min(node_counts)} to {max(node_counts)}"
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"graphs/lab3/{graph_type.replace(' ', '_').lower()}_comparison.png"), dpi=300)
            plt.close()
    
    def run_shortest_path_tests(self, node_counts):
        """Run Dijkstra and Floyd-Warshall tests (Lab 4)"""
        print("\n--- Running Shortest Path Algorithm Analysis ---")
        
        results_dijkstra = {graph_type: [] for graph_type in GRAPH_VARIANTS}
        results_floyd_warshall = {graph_type: [] for graph_type in GRAPH_VARIANTS}

        for size in node_counts:
            print(f"Testing graphs with {size} nodes...")
            for graph_type, generator_func in GRAPH_VARIANTS.items():
                print(f"  {graph_type}...")

                # Generate unweighted graph and convert to weighted
                unweighted_graph = generator_func(size)
                weighted_graph = add_weights(unweighted_graph)

                # Test Dijkstra's algorithm
                try:
                    dijkstra_metrics = measure_dijkstra_performance(weighted_graph, start_node=0)
                    dijkstra_time = dijkstra_metrics["execution_time"]
                except Exception as e:
                    print(f"    ❌ Error in Dijkstra: {e}")
                    dijkstra_time = float('nan')

                # Test Floyd-Warshall algorithm
                try:
                    if size <= 500:  # Limit Floyd-Warshall to smaller graphs due to O(n³) complexity
                        floyd_warshall_metrics = measure_floyd_warshall_performance(weighted_graph)
                        floyd_warshall_time = floyd_warshall_metrics["execution_time"]
                    else:
                        print(f"    ⚠️ Skipping Floyd-Warshall for size {size} (too large)")
                        floyd_warshall_time = float('nan')
                except Exception as e:
                    print(f"    ❌ Error in Floyd-Warshall: {e}")
                    floyd_warshall_time = float('nan')

                # Store results
                results_dijkstra[graph_type].append(dijkstra_time)
                results_floyd_warshall[graph_type].append(floyd_warshall_time)
                print(f"    Dijkstra={format_time(dijkstra_time)}, Floyd-Warshall={format_time(floyd_warshall_time)}")

        # Store the results
        self.performance_data["dijkstra"] = results_dijkstra
        self.performance_data["floyd_warshall"] = results_floyd_warshall
        
        # Plot the results
        self.plot_shortest_path_results(node_counts)
        return results_dijkstra, results_floyd_warshall
    
    def plot_shortest_path_results(self, node_counts):
        """Generate plots for Dijkstra and Floyd-Warshall results"""
        dijkstra_data = self.performance_data["dijkstra"]
        floyd_warshall_data = self.performance_data["floyd_warshall"]
        
        # Create a figure with two vertically stacked subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Dijkstra performance (top)
        for graph_type, times in dijkstra_data.items():
            # Replace NaN values with None for plotting
            times_clean = [t if not np.isnan(t) else None for t in times]
            valid_times = [(s, t) for s, t in zip(node_counts, times_clean) if t is not None]
            if valid_times:
                sizes_valid, times_valid = zip(*valid_times)
                axes[0].plot(sizes_valid, times_valid, marker='o', linewidth=2,
                         label=graph_type, alpha=0.8)

        axes[0].set_title('Dijkstra Algorithm Performance', fontsize=14, weight='bold')
        axes[0].set_xlabel('Number of Nodes', fontsize=12)
        axes[0].set_ylabel('Time (ms)', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper left', framealpha=0.9)

        # Plot 2: Floyd-Warshall performance (bottom)
        for graph_type, times in floyd_warshall_data.items():
            # Replace NaN values with None for plotting
            times_clean = [t if not np.isnan(t) else None for t in times]
            valid_times = [(s, t) for s, t in zip(node_counts, times_clean) if t is not None]
            if valid_times:
                sizes_valid, times_valid = zip(*valid_times)
                axes[1].plot(sizes_valid, times_valid, marker='*', linewidth=2,
                         label=graph_type, alpha=0.8)

        axes[1].set_title('Floyd-Warshall Algorithm Performance', fontsize=14, weight='bold')
        axes[1].set_xlabel('Number of Nodes', fontsize=12)
        axes[1].set_ylabel('Time (ms)', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper left', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "graphs/lab4/pathfinding_algorithm_comparison.png"), dpi=300)
        plt.close()

        # Individual plots for each graph type
        for graph_type in GRAPH_VARIANTS:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get Dijkstra times for this graph type
            dijkstra_times = dijkstra_data[graph_type]
            dijkstra_times_clean = [t if not np.isnan(t) else None for t in dijkstra_times]
            valid_dijkstra = [(s, t) for s, t in zip(node_counts, dijkstra_times_clean) if t is not None]

            # Get Floyd-Warshall times for this graph type
            floyd_warshall_times = floyd_warshall_data[graph_type]
            floyd_warshall_times_clean = [t if not np.isnan(t) else None for t in floyd_warshall_times]
            valid_floyd_warshall = [(s, t) for s, t in zip(node_counts, floyd_warshall_times_clean) if t is not None]

            # Plot Dijkstra times
            if valid_dijkstra:
                sizes_valid, times_valid = zip(*valid_dijkstra)
                ax.plot(sizes_valid, times_valid, marker='o', linewidth=2.5,
                     label='Dijkstra', color=COLOR_SCHEMES['dijkstra'], alpha=0.8)

            # Plot Floyd-Warshall times
            if valid_floyd_warshall:
                sizes_valid, times_valid = zip(*valid_floyd_warshall)
                ax.plot(sizes_valid, times_valid, marker='*', linewidth=2.5,
                     label='Floyd-Warshall', color=COLOR_SCHEMES['floyd_warshall'], alpha=0.8)

            ax.set_title(f'Shortest Path Algorithms on {graph_type}', fontsize=14, weight='bold')
            ax.set_xlabel('Number of Nodes', fontsize=12)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11)
            
            # Add a text box with algorithm details
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
            info_text = f"Graph Type: {graph_type}\nNodes: {min(node_counts)} to {max(node_counts)}"
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"graphs/lab4/{graph_type.replace(' ', '_').lower()}_comparison.png"), dpi=300)
            plt.close()
    
    def run_mst_tests(self, node_counts):
        """Run Kruskal and Prim MST tests (Lab 5)"""
        print("\n--- Running Minimum Spanning Tree Algorithm Analysis ---")
        
        results_kruskal = {graph_type: [] for graph_type in GRAPH_VARIANTS}
        results_prim = {graph_type: [] for graph_type in GRAPH_VARIANTS}

        for size in node_counts:
            print(f"Testing graphs with {size} nodes...")
            for graph_type, generator_func in GRAPH_VARIANTS.items():
                print(f"  {graph_type}...")

                # Generate graph
                graph = generator_func(size)

                # Handle directed acyclic graphs by making them undirected
                if graph_type == "Acyclic Graph":
                    print("    ℹ️ Making acyclic graph undirected for MST")
                    graph = make_undirected(graph)

                # Convert to weighted
                weighted_graph = add_weights(graph)

                # Handle disconnected graphs
                if graph_type == "Disconnected Graph":
                    # Find the largest connected component for testing
                    component = find_largest_component(weighted_graph)
                    print(f"    ℹ️ Using largest component ({len(component)} nodes) of disconnected graph")
                    start_node = component[0] if component else 0
                else:
                    start_node = 0

                # Test Kruskal's algorithm
                try:
                    kruskal_time = test_kruskal(weighted_graph) * 1000  # Convert to ms
                    results_kruskal[graph_type].append(kruskal_time)
                    print(f"    Kruskal: {format_time(kruskal_time)}")
                except Exception as e:
                    print(f"    ❌ Error in Kruskal: {e}")
                    results_kruskal[graph_type].append(float('nan'))

                # Test Prim's algorithm
                try:
                    prim_time = test_prim(weighted_graph, start_node) * 1000  # Convert to ms
                    results_prim[graph_type].append(prim_time)
                    print(f"    Prim: {format_time(prim_time)}")
                except Exception as e:
                    print(f"    ❌ Error in Prim: {e}")
                    results_prim[graph_type].append(float('nan'))

        # Store the results
        self.performance_data["kruskal"] = results_kruskal
        self.performance_data["prim"] = results_prim
        
        # Plot the results
        self.plot_mst_results(node_counts)
        return results_kruskal, results_prim
    
    def plot_mst_results(self, node_counts):
        """Generate plots for Kruskal and Prim results"""
        kruskal_data = self.performance_data["kruskal"]
        prim_data = self.performance_data["prim"]
        
        # Create a figure with two vertically stacked subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Kruskal performance (top)
        for graph_type, times in kruskal_data.items():
            # Handle NaN values for plotting
            times_clean = [t if not np.isnan(t) else None for t in times]
            valid_times = [(s, t) for s, t in zip(node_counts, times_clean) if t is not None]

            if valid_times:
                sizes_valid, times_valid = zip(*valid_times)
                axes[0].plot(sizes_valid, times_valid, marker='o', linewidth=2,
                         label=graph_type, alpha=0.8)

        axes[0].set_title("Kruskal's Algorithm Performance", fontsize=14, weight='bold')
        axes[0].set_xlabel('Number of Nodes', fontsize=12)
        axes[0].set_ylabel('Time (ms)', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper left', framealpha=0.9)

        # Plot 2: Prim performance (bottom)
        for graph_type, times in prim_data.items():
            # Handle NaN values for plotting
            times_clean = [t if not np.isnan(t) else None for t in times]
            valid_times = [(s, t) for s, t in zip(node_counts, times_clean) if t is not None]

            if valid_times:
                sizes_valid, times_valid = zip(*valid_times)
                axes[1].plot(sizes_valid, times_valid, marker='*', linewidth=2,
                         label=graph_type, alpha=0.8)

        axes[1].set_title("Prim's Algorithm Performance", fontsize=14, weight='bold')
        axes[1].set_xlabel('Number of Nodes', fontsize=12)
        axes[1].set_ylabel('Time (ms)', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper left', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "graphs/lab5/spanning_tree_algorithm_comparison.png"), dpi=300)
        plt.close()

        # Individual plots for each graph type
        for graph_type in GRAPH_VARIANTS:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get Kruskal times for this graph type
            kruskal_times = kruskal_data[graph_type]
            kruskal_times_clean = [t if not np.isnan(t) else None for t in kruskal_times]
            valid_kruskal = [(s, t) for s, t in zip(node_counts, kruskal_times_clean) if t is not None]

            # Get Prim times for this graph type
            prim_times = prim_data[graph_type]
            prim_times_clean = [t if not np.isnan(t) else None for t in prim_times]
            valid_prim = [(s, t) for s, t in zip(node_counts, prim_times_clean) if t is not None]

            # Plot Kruskal times
            if valid_kruskal:
                sizes_valid, times_valid = zip(*valid_kruskal)
                ax.plot(sizes_valid, times_valid, marker='o', linewidth=2.5,
                     label='Kruskal', color=COLOR_SCHEMES['kruskal'], alpha=0.8)

            # Plot Prim times
            if valid_prim:
                sizes_valid, times_valid = zip(*valid_prim)
                ax.plot(sizes_valid, times_valid, marker='*', linewidth=2.5,
                     label='Prim', color=COLOR_SCHEMES['prim'], alpha=0.8)

            ax.set_title(f'MST Algorithms on {graph_type}', fontsize=14, weight='bold')
            ax.set_xlabel('Number of Nodes', fontsize=12)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11)
            
            # Add a text box with algorithm details
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
            info_text = f"Graph Type: {graph_type}\nNodes: {min(node_counts)} to {max(node_counts)}"
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"graphs/lab5/{graph_type.replace(' ', '_').lower()}_comparison.png"), dpi=300)
            plt.close()
    
def main():
    """Main function to run all tests"""
    print("\n" + "="*70)
    print("📊 Graph Algorithm Performance Analysis Suite 📊".center(70))
    print("="*70 + "\n")
    
    analyzer = GraphAnalyzer()
    
    # Define sizes for each test type
    traversal_node_counts = [i for i in range(1, 4500, 250)]
    path_finding_node_counts = [10, 50, 100, 200, 220, 300, 400, 500]
    spanning_tree_node_counts = [i for i in range(1, 2400, 150)]
    
    # Ask user which tests to run
    print("\n📋 Select which analysis to run:")
    print("  1️⃣  Graph Traversal (BFS/DFS)")
    print("  2️⃣  Shortest Path (Dijkstra/Floyd-Warshall)")
    print("  3️⃣  Minimum Spanning Tree (Kruskal/Prim)")
    print("  4️⃣  All Analyses")
    choice = input("\n👉 Enter your choice (1-4): ")
    
    start_time = time.time()
    
    # Run selected tests
    if choice == '1' or choice == '4':
        print("\n" + "-"*50)
        print("📈 Running traversal algorithms analysis...")
        analyzer.run_traversal_tests(traversal_node_counts)
        
    if choice == '2' or choice == '4':
        print("\n" + "-"*50)
        print("📈 Running path-finding algorithms analysis...")
        analyzer.run_shortest_path_tests(path_finding_node_counts)
        
    if choice == '3' or choice == '4':
        print("\n" + "-"*50)
        print("📈 Running spanning tree algorithms analysis...")
        analyzer.run_mst_tests(spanning_tree_node_counts)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"✅ All selected analyses completed in {total_time:.2f} seconds!")
    print(f"📁 Results saved to: {os.path.join(analyzer.output_dir, 'graphs/')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()