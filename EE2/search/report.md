# EE2: High acc search

## Objective

To find architectural configs that achieve >95% acc for n=[1,2,3] using periodic boundaries.

## Method

- Progressive complexity (Degree 3-8, Width 4-32, Depth 2-8).
- Periodic boundary conditions.
- > 95% acc.
  >

## Results (Partial)

### Step n=1

- **Config**: deg 3, Width 4, Depth 2.
- **Result**: **100.0%** (Solved immediately).

### Step n = 2

- **Attempted Configs**:
  - Deg 3, W 4, D 2: 13.2%
  - Deg 3, W 8, D 3: 13.2%
  - Deg 5, W 4, D 3: 13.2%
  - Deg 5, W 8, D 4: **Interupted**

**Search Status**: Terminated early due to time constraints.

The search confirmed that n =1 is easily solvable, but n=2 remains difficult even with increased model capacity and periodic boundaries. 

This is consistent with all previous exps and suggests that direct end-to-end learning of $Life^2$ is fundamentally challenging for gradient-based optimization of shallow polynomial networks.
