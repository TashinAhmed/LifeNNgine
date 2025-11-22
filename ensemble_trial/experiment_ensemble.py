import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os

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

def build_small_model(grid_size, filters):
    model = models.Sequential([
        layers.Input(shape=(grid_size, grid_size, 1)),
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

def evaluate_ensemble(models_list, X, y):
    # avg preds
    preds_sum = np.zeros_like(y)
    for model in models_list:
        preds_sum += model.predict(X, verbose=0)
    
    preds_avg = preds_sum / len(models_list)
    preds_binary = (preds_avg > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

# test decreasing filter sizes
filter_sizes = [5, 4, 3, 2]
ENSEMBLE_SIZE = 5

for f in filter_sizes:
    print(f"\n--- Testing Filter Size: {f} ---")
    
    # train single model
    print(f"Training single model with {f} filters...")
    model = build_small_model(GRID_SIZE, f)
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    acc = evaluate_model(model, X_val, y_val)
    print(f"Single Model Accuracy: {acc*100:.2f}%")
    
    if acc < 1.0:
        print(f"Single model failed to reach 100%. Training Ensemble of {ENSEMBLE_SIZE} models...")
        ensemble = []
        # We already have 1, train 4 more
        ensemble.append(model)
        for i in range(ENSEMBLE_SIZE - 1):
            print(f"Training ensemble member {i+2}/{ENSEMBLE_SIZE}...")
            m = build_small_model(GRID_SIZE, f)
            m.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
            ensemble.append(m)
        
        ens_acc = evaluate_ensemble(ensemble, X_val, y_val)
        print(f"Ensemble Accuracy ({f} filters): {ens_acc*100:.2f}%")
        
        if ens_acc > acc:
            print("Ensemble IMPROVED accuracy!")
        if ens_acc > 0.99:
            print("Ensemble achieved 100% (or near)!")
    else:
        print("Single model already achieved 100%.")
