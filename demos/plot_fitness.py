"""
Plot fitness evolution for the EvoMouse simulation.
This script runs a comprehensive evolutionary run (200+ generations) and plots 
the mean and best fitness across generations to visualize convergence behavior.
"""

import matplotlib.pyplot as plt
import numpy as np
from demos.evo_mouse import EvoMouseSimulation


def analyze_convergence(generations, mean_fitness, best_fitness, window_size=20):
    """
    Analyze convergence point by detecting when improvement slows significantly.
    Returns the estimated generation where convergence begins.
    """
    if len(best_fitness) < window_size * 2:
        return None
    
    # Calculate exponential moving average of improvements
    improvements = np.diff(best_fitness)
    ema = np.convolve(np.abs(improvements), np.ones(window_size) / window_size, mode='valid')
    
    # Find where improvement drops significantly
    threshold = np.mean(ema) * 0.5  # 50% of average improvement
    convergence_idx = np.where(ema < threshold)[0]
    
    if len(convergence_idx) > 0:
        return convergence_idx[0] + window_size
    return None


def run_and_plot_fitness():
    """Run the evolution simulation and plot fitness metrics."""
    
    # Create and configure simulation
    sim = EvoMouseSimulation()
    sim.runs = 1
    sim.generations = 200  # Run for 200 generations to observe full convergence
    
    print(f"Running EvoMouse simulation for {sim.generations} generations...")
    print(f"Population size: {sim.population_size}")
    print(f"Timesteps per assessment: {sim.timesteps}")
    print("This may take several minutes. Please wait...\n")
    
    # Run simulation without rendering
    sim.run_simulation(render=False, parallel=False)
    
    # Extract fitness data
    ga = sim.contents["mice"]._genetic_algorithm
    generations = np.arange(len(ga._average_fitness_record))
    mean_fitness = np.array(ga._average_fitness_record)
    best_fitness = np.array(ga._best_fitness_record)
    
    # Analyze convergence
    convergence_point = analyze_convergence(generations, mean_fitness, best_fitness)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Mean and Best Fitness
    ax1.plot(generations, mean_fitness, 'b-', linewidth=2.5, label='Mean Fitness', alpha=0.8)
    ax1.plot(generations, best_fitness, 'r-', linewidth=2.5, label='Best Fitness', alpha=0.8)
    ax1.fill_between(generations, mean_fitness, best_fitness, alpha=0.15, color='purple', label='Fitness Gap')
    
    # Mark convergence point if detected
    if convergence_point:
        ax1.axvline(x=convergence_point, color='green', linestyle='--', linewidth=2, 
                   label=f'Convergence (~Gen {convergence_point})', alpha=0.7)
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness (Cheese Found)', fontsize=12, fontweight='bold')
    ax1.set_title('EvoMouse Evolution: Fitness Progress Over Generations', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(generations)-1)
    
    # Plot 2: Improvement Rate (derivative)
    improvement_mean = np.diff(mean_fitness)
    improvement_best = np.diff(best_fitness)
    ax2.plot(generations[:-1], improvement_mean, 'b-', linewidth=1.5, label='Mean Fitness Change', alpha=0.7)
    ax2.plot(generations[:-1], improvement_best, 'r-', linewidth=1.5, label='Best Fitness Change', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    if convergence_point:
        ax2.axvline(x=convergence_point, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fitness Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Generation-to-Generation Fitness Changes', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, len(generations)-2)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('fitness_evolution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'fitness_evolution.png'")
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"EVOLUTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total Generations: {len(mean_fitness)}")
    print(f"\nInitial Performance:")
    print(f"  Mean Fitness (Gen 1):  {mean_fitness[0]:.6f}")
    print(f"  Best Fitness (Gen 1):  {best_fitness[0]:.6f}")
    print(f"\nFinal Performance:")
    print(f"  Mean Fitness (Gen {len(mean_fitness)}): {mean_fitness[-1]:.6f}")
    print(f"  Best Fitness (Gen {len(mean_fitness)}): {best_fitness[-1]:.6f}")
    print(f"\nImprovement:")
    mean_improvement = (mean_fitness[-1] - mean_fitness[0])
    best_improvement = (best_fitness[-1] - best_fitness[0])
    print(f"  Mean: {mean_improvement:+.6f} ({(mean_improvement/mean_fitness[0]*100):+.1f}%)")
    print(f"  Best: {best_improvement:+.6f} ({(best_improvement/best_fitness[0]*100):+.1f}%)")
    
    if convergence_point:
        print(f"\nConvergence Analysis:")
        print(f"  Estimated convergence point: Generation {convergence_point}")
        print(f"  Best fitness at convergence: {best_fitness[convergence_point]:.6f}")
    
    print(f"{'='*60}\n")
    
    plt.show()


if __name__ == "__main__":
    run_and_plot_fitness()
