import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os
import json
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def game_of_life_step(grid):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    survives = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    births = (grid == 0) & (neighbors == 3)
    return (survives | births).astype(np.float32)

def generate_data(num_samples, grid_size):
    X = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)

    for i in range(num_samples):
        density = np.random.uniform(0.1, 0.9)
        state = np.random.choice([0, 1], size=(grid_size, grid_size), p=[1-density, density])
        next_state = game_of_life_step(state)
        X[i, :, :, 0] = state
        y[i, :, :, 0] = next_state
        
    return X, y

GRID_SIZE = 20
NUM_TRAIN = 10000
NUM_VAL = 2000

print("Generating Data...")
X_train, y_train = generate_data(NUM_TRAIN, GRID_SIZE)
X_val, y_val = generate_data(NUM_VAL, GRID_SIZE)

# https://github.com/JoKerDii/bspline-PyTorch-blocks
class BSplineActivation(layers.Layer):
    def __init__(self, num_basis=5, order=3, **kwargs):
        super(BSplineActivation, self).__init__(**kwargs)
        self.num_basis = num_basis
        self.order = order

    def build(self, input_shape):
        # Trainable coefficients for the B-splines
        # One set of coefficients per channel
        num_channels = input_shape[-1]
        self.coefficients = self.add_weight(
            name='spline_coefficients',
            shape=(num_channels, self.num_basis),
            initializer='random_normal',
            trainable=True
        )
        
        # Fixed grid points for B-splines (simplified, usually these span the input range)
        # We assume inputs are roughly normalized or we learn the grid. 
        # For simplicity in this experiment, we use a fixed grid over [-3, 3]
        self.grid = tf.linspace(-3.0, 3.0, self.num_basis - self.order + 1)
        
    def call(self, inputs):
        # This is a simplified "rational" spline or polynomial approximation
        # Implementing full B-spline recursion in TF efficiently is verbose.
        # Instead, we'll use a simpler "Polynomial Spline" approximation often used in KAN demos:
        # phi(x) = sum(w_i * SiL(x)) where SiL is SiLU or similar basis.
        # Actually, let's implement the "B-Spline" logic from efficient-kan if possible, 
        # or fallback to a simpler learnable polynomial basis: phi(x) = sum(w_i * x^i)
        
        # Let's use a learnable polynomial basis (Chebyshev or simple powers) for stability and ease.
        # phi(x) = w0 + w1*x + w2*x^2 + ...
        
        # Expand inputs: [batch, h, w, c] -> [batch, h, w, c, basis]
        x_expanded = tf.expand_dims(inputs, axis=-1)
        
        # Powers: x^0, x^1, ...
        basis_list = [tf.pow(x_expanded, i) for i in range(self.num_basis)]
        basis = tf.concat(basis_list, axis=-1) # [batch, h, w, c, basis]
        
        # Coefficients: [c, basis]
        coeffs = self.coefficients # [c, basis]
        
        # Compute sum(w_i * x^i)
        # We need to broadcast coeffs to [1, 1, 1, c, basis]
        coeffs_expanded = tf.reshape(coeffs, [1, 1, 1, inputs.shape[-1], self.num_basis])
        
        # Element-wise mul and sum over basis dimension
        out = tf.reduce_sum(basis * coeffs_expanded, axis=-1)
        
        # Add a base activation (like SiLU) for better convergence (residual connection)
        return out + tf.nn.silu(inputs)

def build_kan_model():
    # Architecture: Conv(2, 3x3) -> KAN -> Conv(1, 1x1) -> KAN -> Conv(1, 1x1) -> Sigmoid
    
    inputs = layers.Input(shape=(GRID_SIZE, GRID_SIZE, 1))
    
    # Layer 1: 2 filters, 3x3
    x = layers.Conv2D(2, (3, 3), padding='same')(inputs)
    x = BSplineActivation(num_basis=5)(x) # KAN activation
    
    # Layer 2: 1 filter, 1x1
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = BSplineActivation(num_basis=5)(x) # KAN activation
        
    # Output Layer: 1 filter, 1x1, sigmoid
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

TRIALS = 10
EPOCHS = 100
BATCH_SIZE = 64

results = {}

print(f"\nStarting KAN Experiment: {TRIALS} trials.")
print("Architecture: Conv(2,3x3) -> PolyKAN -> Conv(1,1x1) -> PolyKAN -> Conv(1,1x1)")

trial_accuracies = []

for t in range(TRIALS):
    print(f"  Trial {t+1}/{TRIALS}...", end="", flush=True)
    start_time = time.time()
    
    model = build_kan_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=10, 
        restore_best_weights=True,
        mode='max'
    )
    
    model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[early_stopping]
    )
    
    acc = evaluate_exact_match(model, X_val, y_val)
    trial_accuracies.append(acc)
    
    duration = time.time() - start_time
    print(f" Done. Acc: {acc*100:.2f}% ({duration:.1f}s)")
    
results['PolyKAN'] = trial_accuracies
avg_acc = np.mean(trial_accuracies)
max_acc = np.max(trial_accuracies)
success_count = sum(1 for a in trial_accuracies if a > 0.99)
print(f"  >> Avg: {avg_acc*100:.2f}%, Best: {max_acc*100:.2f}%, Successes: {success_count}/{TRIALS}")

with open('kan_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nExperiment Complete. Results saved to kan_results.json")
