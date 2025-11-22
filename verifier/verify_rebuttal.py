import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os

# Suppress TF logs
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
NUM_TRAIN = 1000
NUM_VAL = 200

print("Generating Data...")
X_train, y_train = generate_data(NUM_TRAIN, GRID_SIZE)
X_val, y_val = generate_data(NUM_VAL, GRID_SIZE)

def build_model(grid_size):
    model = models.Sequential([
        layers.Input(shape=(grid_size, grid_size, 1)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("Building Model...")
model = build_model(GRID_SIZE)

print("Training...")
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

print("Evaluating...")
preds = model.predict(X_val)
preds_binary = (preds > 0.5).astype(np.float32)
matches = np.all(preds_binary == y_val, axis=(1, 2, 3))
exact_match_acc = np.mean(matches)

print(f"Exact Match Accuracy: {exact_match_acc * 100:.2f}%")

if exact_match_acc > 0.9:
    print("VERIFICATION SUCCESS")
else:
    print("VERIFICATION FAILURE")
