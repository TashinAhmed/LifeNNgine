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

class QuadraticActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(QuadraticActivation, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable coefficients for the quadratic polynomial: w2*x^2 + w1*x + w0
        # One set of coefficients per channel
        num_channels = input_shape[-1]
        
        self.w2 = self.add_weight(
            name='w2',
            shape=(1, 1, 1, num_channels),
            initializer='zeros', # Start with 0 curvature
            trainable=True
        )
        self.w1 = self.add_weight(
            name='w1',
            shape=(1, 1, 1, num_channels),
            initializer='ones', # Start with identity slope
            trainable=True
        )
        self.w0 = self.add_weight(
            name='w0',
            shape=(1, 1, 1, num_channels),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        # f(x) = w2*x^2 + w1*x + w0
        # We add a base activation (SiLU) as a residual to help stability, 
        # or we can just rely on the polynomial.
        # Given the user asked for "2 degree KAN", let's try pure polynomial + residual identity.
        # Actually, PReLU is identity + modification.
        # Let's do: output = w2*x^2 + w1*x + w0
        # Initialized to identity (w2=0, w1=1, w0=0)
        
        return self.w2 * tf.square(inputs) + self.w1 * inputs + self.w0

def build_kan_degree2_model():
    # Architecture: Conv(2, 3x3) -> QuadKAN -> Conv(1, 1x1) -> QuadKAN -> Conv(1, 1x1) -> Sigmoid
    
    inputs = layers.Input(shape=(GRID_SIZE, GRID_SIZE, 1))
    
    # Layer 1: 2 filters, 3x3
    x = layers.Conv2D(2, (3, 3), padding='same')(inputs)
    x = QuadraticActivation()(x) 
    
    # Layer 2: 1 filter, 1x1
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = QuadraticActivation()(x)
        
    # Output Layer: 1 filter, 1x1, sigmoid
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

# config
TRIALS = 10
EPOCHS = 100
BATCH_SIZE = 64

results = {}

print(f"\nStarting KAN Degree 2 Experiment: {TRIALS} trials.")
print("Architecture: Conv(2,3x3) -> Quadratic -> Conv(1,1x1) -> Quadratic -> Conv(1,1x1)")

trial_accuracies = []

for t in range(TRIALS):
    print(f"  Trial {t+1}/{TRIALS}...", end="", flush=True)
    start_time = time.time()
    
    model = build_kan_degree2_model()
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
    
results['QuadraticKAN'] = trial_accuracies
avg_acc = np.mean(trial_accuracies)
max_acc = np.max(trial_accuracies)
success_count = sum(1 for a in trial_accuracies if a > 0.99)
print(f"  >> Avg: {avg_acc*100:.2f}%, Best: {max_acc*100:.2f}%, Successes: {success_count}/{TRIALS}")

with open('kan_degree2_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nExperiment Complete. Results saved to kan_degree2_results.json")
