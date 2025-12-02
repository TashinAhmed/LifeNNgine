import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def life_step(grid, rule_b, rule_s):
    """
    Generic Life-like cellular automaton step.
    rule_b: list of neighbor counts for birth (e.g., [3])
    rule_s: list of neighbor counts for survival (e.g., [2, 3])
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    birth_mask = np.isin(neighbors, rule_b)
    survive_mask = np.isin(neighbors, rule_s)
    survives = (grid == 1) & survive_mask
    births = (grid == 0) & birth_mask
    return (survives | births).astype(np.float32)

def generate_data(num_samples, grid_size, rule_b, rule_s):
    X = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)
    for i in range(num_samples):
        density = np.random.uniform(0.1, 0.9)
        state = np.random.choice([0, 1], size=(grid_size, grid_size), p=[1-density, density])
        next_state = life_step(state, rule_b, rule_s)
        X[i, :, :, 0] = state
        y[i, :, :, 0] = next_state
    return X, y

class QuadraticActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(QuadraticActivation, self).__init__(**kwargs)
    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.w2 = self.add_weight(name='w2', shape=(1, 1, 1, num_channels), initializer='zeros', trainable=True)
        self.w1 = self.add_weight(name='w1', shape=(1, 1, 1, num_channels), initializer='ones', trainable=True)
        self.w0 = self.add_weight(name='w0', shape=(1, 1, 1, num_channels), initializer='zeros', trainable=True)
    def call(self, inputs):
        return self.w2 * tf.square(inputs) + self.w1 * inputs + self.w0

def build_kan_model(grid_size, filters=2):
    inputs = layers.Input(shape=(grid_size, grid_size, 1))
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = QuadraticActivation(name='quad1')(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = QuadraticActivation(name='quad2')(x)
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

def visualize_functions(model, title, filename):
    """Visualizes the learned quadratic functions."""
    quad1 = model.get_layer('quad1')
    quad2 = model.get_layer('quad2')    
    w2_1 = quad1.w2.numpy().flatten()
    w1_1 = quad1.w1.numpy().flatten()
    w0_1 = quad1.w0.numpy().flatten()
    w2_2 = quad2.w2.numpy().flatten()
    w1_2 = quad2.w1.numpy().flatten()
    w0_2 = quad2.w0.numpy().flatten()
    x = np.linspace(-10, 10, 200)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(len(w2_1)):
        y = w2_1[i] * x**2 + w1_1[i] * x + w0_1[i]
        plt.plot(x, y, label=f'Filter {i+1}')
    plt.title(f'{title} - Layer 1 Activations')
    plt.xlabel('Input (Convolution Output)')
    plt.ylabel('Output (Activation)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    y = w2_2[0] * x**2 + w1_2[0] * x + w0_2[0]
    plt.plot(x, y, label=f'Filter 1', color='red')
    plt.title(f'{title} - Layer 2 Activation')
    plt.xlabel('Input (Convolution Output)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved function plot to {filename}")

def visualize_dynamics(model, X_sample, title, filename):
    """Visualizes the feature maps at each layer for a sample input."""
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(X_sample[np.newaxis, ...], verbose=0)
    
    # Identify layers of interest
    # 0: Input
    # 1: Conv1
    # 2: Quad1
    # 3: Conv2
    # 4: Quad2
    # 5: Output (Sigmoid)
    
    input_grid = activations[0][0, :, :, 0]
    conv1_out = activations[1][0, :, :, :]
    quad1_out = activations[2][0, :, :, :]
    conv2_out = activations[3][0, :, :, 0]
    quad2_out = activations[4][0, :, :, 0]
    final_out = activations[5][0, :, :, 0]
    
    num_filters = conv1_out.shape[-1]
    
    plt.figure(figsize=(15, 8))
    
    # i/p
    plt.subplot(2, num_filters + 2, 1)
    plt.imshow(input_grid, cmap='binary', vmin=0, vmax=1)
    plt.title("Input Grid")
    plt.axis('off')
    
    for i in range(num_filters):
        plt.subplot(2, num_filters + 2, 2 + i)
        plt.imshow(conv1_out[:, :, i], cmap='viridis')
        plt.title(f"L1 Conv F{i+1}")
        plt.axis('off')
        plt.subplot(2, num_filters + 2, 2 + num_filters + 2 + i)
        plt.imshow(quad1_out[:, :, i], cmap='viridis')
        plt.title(f"L1 Quad F{i+1}")
        plt.axis('off')
    plt.subplot(2, num_filters + 2, num_filters + 2)
    plt.imshow(conv2_out, cmap='viridis')
    plt.title("L2 Conv")
    plt.axis('off')
    
    plt.subplot(2, num_filters + 2, 2 * (num_filters + 2))
    plt.imshow(quad2_out, cmap='viridis')
    plt.title("L2 Quad")
    plt.axis('off')
    
    plt.subplot(2, num_filters + 2, num_filters + 3)
    plt.imshow(final_out, cmap='binary', vmin=0, vmax=1)
    plt.title("Final Output")
    plt.axis('off')
    
    plt.suptitle(f"{title} - Network Dynamics")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved dynamics plot to {filename}")

GRID_SIZE = 20
NUM_TRAIN = 5000
NUM_VAL = 1000

# Experiment A: GoL (B3/S23)
print("\n=== Experiment A: Conway's Game of Life (B3/S23) ===")
X_train_gol, y_train_gol = generate_data(NUM_TRAIN, GRID_SIZE, [3], [2, 3])
X_val_gol, y_val_gol = generate_data(NUM_VAL, GRID_SIZE, [3], [2, 3])

model_gol = build_kan_model(GRID_SIZE, filters=2)
model_gol.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training GoL Model...")
model_gol.fit(X_train_gol, y_train_gol, epochs=30, batch_size=64, verbose=0)
acc_gol = evaluate_exact_match(model_gol, X_val_gol, y_val_gol)
print(f"GoL Accuracy: {acc_gol*100:.2f}%")

if acc_gol > 0.99:
    visualize_functions(model_gol, "GoL (B3/S23)", "analysis_gol_functions.png")
    visualize_dynamics(model_gol, X_val_gol[0], "GoL (B3/S23)", "analysis_gol_dynamics.png")
else:
    print("GoL model failed to converge, skipping visualization.")


# Experiment B: Rule (B368/S245)
print("\n=== Experiment B: Complex Rule (B368/S245) ===")
# This rule has multiple disjoint intervals: Birth {3, 6, 8}, Survive {2, 4, 5}
X_train_complex, y_train_complex = generate_data(NUM_TRAIN, GRID_SIZE, [3, 6, 8], [2, 4, 5])
X_val_complex, y_val_complex = generate_data(NUM_VAL, GRID_SIZE, [3, 6, 8], [2, 4, 5])

# Trial 1: Minimal Architecture (2 filters)
print("Testing Minimal Architecture (2 filters)...")
model_complex_2 = build_kan_model(GRID_SIZE, filters=2)
model_complex_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_complex_2.fit(X_train_complex, y_train_complex, epochs=50, batch_size=64, verbose=0)
acc_complex_2 = evaluate_exact_match(model_complex_2, X_val_complex, y_val_complex)
print(f"Complex Rule (2 filters) Accuracy: {acc_complex_2*100:.2f}%")

# Trial 2: Increased Channels (4 filters)
print("Testing Increased Channels (4 filters)...")
model_complex_4 = build_kan_model(GRID_SIZE, filters=4)
model_complex_4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_complex_4.fit(X_train_complex, y_train_complex, epochs=50, batch_size=64, verbose=0)
acc_complex_4 = evaluate_exact_match(model_complex_4, X_val_complex, y_val_complex)
print(f"Complex Rule (4 filters) Accuracy: {acc_complex_4*100:.2f}%")

if acc_complex_4 > 0.99:
    visualize_functions(model_complex_4, "Complex (B368/S245) - 4 Filters", "analysis_complex_functions.png")
elif acc_complex_2 > 0.99:
    visualize_functions(model_complex_2, "Complex (B368/S245) - 2 Filters", "analysis_complex_functions.png")

print("\nAnalysis Complete.")
