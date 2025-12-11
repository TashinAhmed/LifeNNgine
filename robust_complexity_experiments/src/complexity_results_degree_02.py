
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def life_step(grid, rule_b, rule_s):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    birth_mask = np.isin(neighbors, rule_b)
    survive_mask = np.isin(neighbors, rule_s)
    return ((grid == 1) & survive_mask) | ((grid == 0) & birth_mask)

def generate_data(num_samples, grid_size, rule_b, rule_s):
    X = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)
    for i in range(num_samples):
        density = np.random.uniform(0.1, 0.9)
        state = np.random.choice([0, 1], size=(grid_size, grid_size), p=[1-density, density])
        X[i, :, :, 0] = state
        y[i, :, :, 0] = life_step(state, rule_b, rule_s)
    return X, y

class PolynomialActivation(layers.Layer):
    def __init__(self, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.coeffs = []
        for i in range(self.degree + 1):
            init = 'ones' if i == 1 else 'zeros'
            self.coeffs.append(self.add_weight(name=f'w{i}', shape=(1, 1, 1, num_channels), initializer=init))

    def call(self, inputs):
        out = self.coeffs[0]
        x_pow = inputs
        out += self.coeffs[1] * x_pow
        for i in range(2, self.degree + 1):
            x_pow *= inputs
            out += self.coeffs[i] * x_pow
        return out

    def get_config(self):
        return {**super().get_config(), "degree": self.degree}

def build_model(grid_size, degree, filters):
    inputs = layers.Input(shape=(grid_size, grid_size, 1))
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = PolynomialActivation(degree=degree)(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = PolynomialActivation(degree=degree)(x)
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    return models.Model(inputs, outputs)

def run_robust_01():
    GRID_SIZE = 20
    NUM_TRAIN = 5000
    NUM_VAL = 1000
    NUM_RUNS = 10
    
    rules = {
        "Game of Life": ([3], [2, 3]),
        "Replicator": ([1,3,5,7], [1,3,5,7]),
        "Fredkin": ([1,3,5,7], [0,2,4,6,8])
    }
    degrees = [2, 4, 8, 16]
    
    md_output = "# Robust Experiment 1 Results (10 Runs)\n\n"
    md_output += "| Rule | Degree | Mean Acc | Std Dev | Success Rate (100%) |\n"
    md_output += "|---|---|---|---|---|\n"
    
    for rule_name, (b, s) in rules.items():
        print(f"=== {rule_name} ===")
        
        for deg in degrees:
            accuracies = []
            start_time = time.time()
            print(f"  Degree {deg}: ", end="", flush=True)
            
            for run_idx in range(NUM_RUNS):
                X_train, y_train = generate_data(NUM_TRAIN, GRID_SIZE, b, s)
                X_val, y_val = generate_data(NUM_VAL, GRID_SIZE, b, s) # New val set each time
                
                model = build_model(GRID_SIZE, deg, filters=4)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
                
                preds = (model.predict(X_val, verbose=0) > 0.5).astype(np.float32)
                acc = np.mean(np.all(preds == y_val, axis=(1,2,3)))
                accuracies.append(acc)
                print(f".", end="", flush=True)
                
            mean_acc = np.mean(accuracies) * 100
            std_acc = np.std(accuracies) * 100
            success_rate = np.mean(np.array(accuracies) > 0.99) * 100
            
            print(f" Mean: {mean_acc:.1f}%")
            md_output += f"| {rule_name} | {deg} | {mean_acc:.1f}% | {std_acc:.2f}% | {success_rate:.0f}% |\n"

    with open("robust_results_degree.md", "w") as f:
        f.write(md_output)
    print("Saved results to robust_results_degree.md")

if __name__ == "__main__":
    run_robust_01()
