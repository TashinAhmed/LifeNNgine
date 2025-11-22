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

def build_model(grid_size, filters):
    model = models.Sequential([
        layers.Input(shape=(grid_size, grid_size, 1)),
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

# cofig
FILTER_SIZES = [16, 12, 10, 8, 6, 5, 4]
TRIALS = 5
EPOCHS = 50
BATCH_SIZE = 64

results = {}

print(f"\nStarting Experiment: {TRIALS} trials per filter size.")
print(f"Filter Sizes: {FILTER_SIZES}")

for f in FILTER_SIZES:
    print(f"\n--- Testing Filter Size: {f} ---")
    trial_accuracies = []
    
    for t in range(TRIALS):
        print(f"  Trial {t+1}/{TRIALS}...", end="", flush=True)
        start_time = time.time()
        
        model = build_model(GRID_SIZE, f)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=5, 
            restore_best_weights=True,
            mode='max'
        )
        
        history = model.fit(
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
        
    results[f] = trial_accuracies
    avg_acc = np.mean(trial_accuracies)
    max_acc = np.max(trial_accuracies)
    print(f"  >> Average: {avg_acc*100:.2f}%, Best: {max_acc*100:.2f}%")

with open('single_experiment_data.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nExperiment Complete. Results saved to single_experiment_data.json")
