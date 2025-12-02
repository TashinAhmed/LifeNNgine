# KAN Degree 2 Analysis & Complex Rules

Performed a deeper analysis of the successful 2-deg KAN (Quadratic) model \& tested its limits on a more complex Lifelike rule.

## 1. Why Quadratic Activation Works for GoL
Visualized the learned functions of the successful model (`analysis_gol_functions.png`).
- **Mechanism**: The model learns parabolas that act as **band-pass filters**.
- **The "Bump"**: A parabola $y = ax^2 + bx + c$ with $a > 0$ (opening up) or $a < 0$ (opening down) can easily create a "bump" or "dip" over a specific range of input values.
- **Game of Life Logic**:
    - "Survive if neighbors $\in \{2, 3\}$".
    - "Birth if neighbors $\in \{3\}$".
    - These are simple intervals. A single parabola can easily output a high value for inputs in $[2, 3]$ and low values elsewhere (or vice versa).
- **ReLU Comparison**: A ReLU is piecewise linear. To create a **bump** (up then down), you need at least **two** ReLU neurons (one to go up, one to go down) and subtract them. This explains why the minimal architecture (2 filters) struggled with ReLU it barely had enough **parts** to build the necessary logic. Quadratic activation provides this **bump** capability natively in a single neuron.

## 2. B368/S245 Experiment
Tested the same architecture on a much more complex rule: **B368/S245**.
- **Birth**: 3, 6, 8 neighbors.
- **Survive**: 2, 4, 5 neighbors.
- **Logic**: This requires capturing multiple disjoint intervals: $\{2\}, \{3\}, \{4, 5\}, \{6\}, \{8\}$. This is a highly non monotonic, **wavy** function.

### Results
| Arch | acc | res |
| :--- | :--- | :--- |
| **2 Filters (Quad)** | 2.00% | **Failed** |
| **4 Filters (Quad)** | 2.00% | **Faled** |

### Interpretation
- **Capacity Mismatch**: A single parabola (deg 2) can only change direction once. It cannot wiggle up and down enough to capture the alternating **good** and **bad** neighbor counts of B368/S245.
- **Need for Higher Degree/More Channels**: To approximate this complex function, we would need:
    - **Higher Degree**: A polynomial of degree ~6 or ~8 to have enough **wiggles**.
    - **More Channels**: Summing many parabolas (e.g., 8+ filters) to approximate the complex shape.
- **Conclusion**: The **complexity** of the CA rule directly dictates the required capacity (degree or width) of the KAN. Minimal QuadKAN is perfect for simple rules like GoL, but insufficient for highly complex ones.

## 3. Visualizations
Generated the following plots in the `rebuttal` directory:
- `analysis_gol_functions.png`: Shows the learned parabolas for the successful GoL model.
- `analysis_gol_dynamics.png`: Shows the feature maps at each layer, allowing us to watch the network "think".
