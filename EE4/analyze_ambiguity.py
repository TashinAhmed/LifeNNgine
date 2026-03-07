import numpy as np
import itertools

def analyze_rule_ambiguity(neighbor_counts_seen):
    """
    Given a set of (cell_state, neighbor_sum, next_state) observations,
    calculate how many Life-like rules (out of 2^18 total) are consistent.
    
    Life-like rules:
    Birth: subset of {0..8} (9 bits)
    Survival: subset of {0..8} (9 bits)
    Total 18 bits = 262,144 rules.
    """
    # Initialize constraints:
    # 0: must be 0
    # 1: must be 1
    # -1: unknown
    birth_constraints = np.full(9, -1)
    survival_constraints = np.full(9, -1)
    
    for state, neighbors, next_state in neighbor_counts_seen:
        if state == 0:
            if birth_constraints[neighbors] != -1 and birth_constraints[neighbors] != next_state:
                return 0 # Contradiction! (Shouldn't happen for GoL data)
            birth_constraints[neighbors] = next_state
        else:
            if survival_constraints[neighbors] != -1 and survival_constraints[neighbors] != next_state:
                return 0 # Contradiction!
            survival_constraints[neighbors] = next_state
            
    # Count free bits
    free_birth = np.sum(birth_constraints == -1)
    free_survival = np.sum(survival_constraints == -1)
    
    num_consistent = 2**(free_birth + free_survival)
    return num_consistent

def collect_observations_from_grid(grid, engine, rule):
    # This is more involved if we want to extract from arbitrary grid.
    # For now, let's just simulate what happens with random vs bespoke.
    pass

def run_analysis():
    print("--- Rule Ambiguity Analysis ---")
    
    # 1. Total possible rules
    print(f"Total Life-like rules: {2**18:,}")
    
    # 2. Bespoke Grid (512 configs)
    # The bespoke grid covers ALL combinations of (state, neighbors)
    # So it should uniquely identify the rule?
    # Actually, a single (state, neighbor_sum) pair only fixes ONE bit of the 18.
    # Since neighbor sums 0-8 happen for both state 0 and 1, we need all 18 pairs.
    # A random grid might miss some rare neighbor counts (like 8).
    
    counts = []
    # Game of Life: B3 / S23
    for s in [0, 1]:
        for n in range(9):
            if s == 0:
                next_s = 1 if n == 3 else 0
            else:
                next_s = 1 if n in [2, 3] else 0
            counts.append((s, n, next_s))
            
    # If we see all 18 combinations:
    num_bespoke = analyze_rule_ambiguity(counts)
    print(f"Ambiguity with all 18 (state, neighbors) pairs: {num_bespoke} (Rule uniquely identified)")
    
    # 3. Random data simulation
    # How many pairs do we see on average in a 10x10 random grid?
    print("\nSimulation of Random Grid (20x20, density 0.3):")
    for trial in range(3):
        grid = (np.random.rand(20, 20) < 0.3).astype(int)
        # Count neighbors
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        from scipy.signal import convolve2d
        neighbors = convolve2d(grid, kernel, mode='same', boundary='wrap')
        
        seen = set()
        for r in range(20):
            for c in range(20):
                s = grid[r, c]
                n = neighbors[r, c]
                # GoL logic
                if s == 0: next_s = 1 if n == 3 else 0
                else: next_s = 1 if n in [2, 3] else 0
                seen.add((s, n, next_s))
        
        num_random = analyze_rule_ambiguity(list(seen))
        print(f"Trial {trial+1}: Saw {len(seen)}/18 pairs. Consistent rules: {num_random:,}")

if __name__ == "__main__":
    run_analysis()
