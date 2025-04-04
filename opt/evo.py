import pygad
import numpy as np
import time
import csv
from datetime import datetime, timedelta

from src import utils, birdnet
from brute import calc_dif

# Define parameter mappings (from gene values to actual parameter values)
param_config = {
    'denoise_method': {
        'type': 'categorical',
        'values': ['dwt', 'wpt', 'emd'],
        'gene_range': (0, 2)  # Integer range for gene (will be rounded and used as index)
    },
    'n_noise_layers': {
        'type': 'integer',
        'range': (1, 7)  # Min, Max (inclusive)
    },
    'wavelet': {
        'type': 'categorical',
        'values': ['bior4.4', 'dmey', 'db6', 'db4', 'db8', 'sym8', 'coif3'],
        'gene_range': (0, 6)  # Integer range for gene (will be rounded and used as index)
    },
    'level': {
        'type': 'integer',
        'range': (1, 7)  # Min, Max (inclusive)
    },
    'threshold_method': {
        'type': 'categorical',
        'values': ['soft', 'hard'],
        'gene_range': (0, 1)  # Integer range for gene (will be rounded and used as index)
    },
    'threshold_factor': {
        'type': 'float',
        'range': (0.1, 1.0)  # Min, Max
    },
    'denoise_strength': {
        'type': 'float',
        'range': (0.1, 2.0)  # Min, Max
    },
    'preserve_ratio': {
        'type': 'float',
        'range': (0.5, 0.95)  # Min, Max
    }
}

# Extract gene spaces for PyGAD correctly
gene_spaces = []
for param, config in param_config.items():
    if config['type'] == 'categorical':
        # For categorical, create a list of all possible integer values in the range
        min_val, max_val = config['gene_range']
        gene_spaces.append(list(range(min_val, max_val + 1)))
    elif config['type'] == 'integer':
        # For integer, create a list of all possible values
        min_val, max_val = config['range']
        gene_spaces.append(list(range(min_val, max_val + 1)))
    else:  # float
        # For float parameters, create a properly spaced range with many values
        min_val, max_val = config['range']
        # Create 100 evenly spaced values in the range
        values = np.linspace(min_val, max_val, 100).tolist()
        gene_spaces.append(values)


# Function to convert a gene to actual parameter value
def gene_to_param_value(gene_value, param_name):
    config = param_config[param_name]

    if config['type'] == 'categorical':
        # Convert gene value to index (rounding to nearest integer)
        index = round(gene_value)
        # Ensure index is within bounds
        index = max(0, min(index, len(config['values']) - 1))
        return config['values'][index]
    elif config['type'] == 'integer':
        # Round to nearest integer and ensure within bounds
        value = round(gene_value)
        return max(config['range'][0], min(value, config['range'][1]))
    else:  # float
        # Ensure within bounds
        return max(config['range'][0], min(gene_value, config['range'][1]))


# Convert a complete solution (chromosome) to parameter dictionary
def solution_to_params(solution):
    params = {}
    for i, (param_name, _) in enumerate(param_config.items()):
        params[param_name] = gene_to_param_value(solution[i], param_name)
    return params


# Results tracking
best_solution = None
best_fitness = -float('inf')
start_time = None
results = []
eval_log_file = None  # Will hold the file handle for real-time logging


# Fitness function
def fitness_func(ga_instance, solution, solution_idx):
    global best_solution, best_fitness, results, eval_log_file

    # Convert solution to actual parameters
    params = solution_to_params(solution)

    # Time this evaluation
    eval_start = time.time()

    # Call the calc_dif function with these parameters
    result = calc_dif(**params)

    # Track execution time
    eval_time = time.time() - eval_start

    # Create evaluation record
    eval_record = {
        'params': params,
        'fitness': result,
        'eval_time': eval_time,
        'generation': ga_instance.generations_completed,
        'evaluation': len(results) + 1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    }

    # Store the result
    results.append(eval_record)

    # Log immediately to file
    if eval_log_file:
        # Create string with all parameters and results
        log_entry = f"{eval_record['timestamp']},{eval_record['generation']},{eval_record['evaluation']},{result},{eval_time}"
        for k, v in params.items():
            log_entry += f",{v}"
        eval_log_file.write(log_entry + "\n")
        eval_log_file.flush()  # Ensure it's written immediately

    # Update best solution if needed
    if result > best_fitness:
        best_fitness = result
        best_solution = params.copy()

        # Print the new best
        print(f"\nNew best solution found (gen {ga_instance.generations_completed}):")
        print(f"Fitness: {best_fitness}")
        print("Parameters:")
        for k, v in best_solution.items():
            print(f"  {k}: {v}")
        print("-----------------------------------")

    return result


# On generation callback - print progress
def on_generation(ga_instance):
    global start_time

    # Calculate elapsed time and metrics
    current_time = time.time()
    elapsed = current_time - start_time
    gen = ga_instance.generations_completed

    # Print progress every generation
    if gen % 1 == 0:
        # Get the fitness values from the population fitness array, not from solution objects
        avg_fitness = np.mean(ga_instance.last_generation_fitness)
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if gen > 0:
            time_per_gen = elapsed / gen
            gens_remaining = ga_instance.num_generations - gen
            remaining = time_per_gen * gens_remaining
            remaining_str = str(timedelta(seconds=int(remaining)))
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_str = eta.strftime('%H:%M:%S')

            print(f"Generation {gen}/{ga_instance.num_generations} - " +
                  f"Best: {ga_instance.best_solution()[1]:.4f}, " +
                  f"Avg: {avg_fitness:.4f}, " +
                  f"Elapsed: {elapsed_str}, " +
                  f"ETA: {eta_str} (remaining: {remaining_str})")
        else:
            print(f"Generation {gen}/{ga_instance.num_generations} - " +
                  f"Best: {ga_instance.best_solution()[1]:.4f}, " +
                  f"Avg: {avg_fitness:.4f}, " +
                  f"Elapsed: {elapsed_str}")


def main():
    global start_time, results, eval_log_file

    # PyGAD parameters
    num_generations = 50
    num_parents_mating = 5
    sol_per_pop = 50
    parent_selection_type = "tournament"
    K_tournament = 3
    crossover_type = "single_point"
    mutation_type = "adaptive"

    # FIX: Use a list for mutation_percent_genes
    # First value is initial mutation rate, second is final mutation rate
    mutation_percent_genes = [10, 2]  # Start with 10% mutation, end with 2%

    # Create real-time evaluation log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_filename = f"evo/pygad_calc_dif_evals_{timestamp}.csv"

    # Open the file for writing
    eval_log_file = open(eval_log_filename, 'w', newline='')

    # Write header
    header = "timestamp,generation,evaluation,fitness,eval_time"
    for param in param_config.keys():
        header += f",{param}"
    eval_log_file.write(header + "\n")
    eval_log_file.flush()

    # Create the GA instance
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=len(param_config),
        gene_space=gene_spaces,  # Use the correctly defined gene spaces
        parent_selection_type=parent_selection_type,
        K_tournament=K_tournament,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        on_generation=on_generation,
        save_best_solutions=False,
        initial_population=None
    )

    print("Starting PyGAD optimization for calc_dif")
    print(f"Population size: {sol_per_pop}, Generations: {num_generations}")
    print(f"Total evaluations: {sol_per_pop * num_generations}")
    print("Parameter space:")
    for param, config in param_config.items():
        if config['type'] == 'categorical':
            print(f"  {param}: {config['values']}")
        else:
            print(f"  {param}: {config['range']}")

    # Start timing
    start_time = time.time()

    # Run the genetic algorithm
    ga_instance.run()

    # Get the final best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    final_params = solution_to_params(solution)

    # Print results
    print("\nOptimization complete!")
    print(f"Best fitness: {solution_fitness}")
    print("Best parameters:")
    for k, v in final_params.items():
        print(f"  {k}: {v}")

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"pygad_optimization_results_{timestamp}.csv"

    with open(results_file, 'w', newline='') as csvfile:
        # Determine all fields needed for the CSV
        all_param_keys = list(param_config.keys())
        fieldnames = ['generation', 'evaluation'] + all_param_keys + ['fitness', 'eval_time']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Group results by generation
        for gen_num in range(num_generations):
            gen_results = [r for r in results if r['generation'] == gen_num]

            # Write each evaluation in this generation
            for eval_num, result in enumerate(gen_results):
                row = {
                    'generation': gen_num,
                    'evaluation': eval_num,
                    'fitness': result['fitness'],
                    'eval_time': result['eval_time']
                }

                # Add all parameter values
                for param, value in result['params'].items():
                    row[param] = value

                writer.writerow(row)

    print(f"Results saved to: {results_file}")

    # Total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {timedelta(seconds=int(total_time))}")

    # Close the evaluation log file
    if eval_log_file:
        eval_log_file.close()
        print(f"All individual calc_dif evaluations logged to: {eval_log_filename}")

    # Plot the convergence curve
    try:
        ga_instance.plot_fitness(title="PyGAD Optimization for calc_dif")
    except:
        print("Unable to create convergence plot. You may need matplotlib installed.")

    return final_params, solution_fitness


if __name__ == "__main__":
    utils.TRAIN_DIR = "/home/mikhail/prj/bc_25_data/train_audio"
    birdnet.CSV_PATH = "/home/mikhail/prj/bc_25_data/taxonomy.csv"

    main()