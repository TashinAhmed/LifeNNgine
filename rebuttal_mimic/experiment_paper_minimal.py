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

def build_paper_minimal_model(activation='relu'):
    # Exact architecture from paper:
    # Conv(2, 3x3) -> Conv(1, 1x1) -> Conv(1, 1x1)
    
    inputs = layers.Input(shape=(GRID_SIZE, GRID_SIZE, 1))
    
    # Layer 1: 2 filters, 3x3
    if activation == 'prelu':
        x = layers.Conv2D(2, (3, 3), padding='same')(inputs)
        x = layers.PReLU()(x)
    else:
        x = layers.Conv2D(2, (3, 3), padding='same', activation=activation)(inputs)
        
    # Layer 2: 1 filter, 1x1
    if activation == 'prelu':
        x = layers.Conv2D(1, (1, 1), padding='same')(x)
        x = layers.PReLU()(x)
    else:
        x = layers.Conv2D(1, (1, 1), padding='same', activation=activation)(x)
        
    # Output Layer: 1 filter, 1x1, sigmoid
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

# cofig
TRIALS = 10
EPOCHS = 100
BATCH_SIZE = 64

variations = [
    {'name': 'Baseline (ReLU + BCE)', 'act': 'relu', 'loss': 'binary_crossentropy'},
    {'name': 'ReLU + MSE', 'act': 'relu', 'loss': 'mse'},
    {'name': 'Swish + BCE', 'act': 'swish', 'loss': 'binary_crossentropy'},
    {'name': 'GeLU + BCE', 'act': 'gelu', 'loss': 'binary_crossentropy'},
    {'name': 'PReLU + BCE', 'act': 'prelu', 'loss': 'binary_crossentropy'},
]

results = {}

print(f"\nStarting Paper Rebuttal Experiment: {TRIALS} trials per variation.")
print("Architecture: Conv(2,3x3) -> Conv(1,1x1) -> Conv(1,1x1)")

for v in variations:
    name = v['name']
    act = v['act']
    loss = v['loss']
    
    print(f"\n--- Testing Variation: {name} ---")
    trial_accuracies = []
    
    for t in range(TRIALS):
        print(f"  Trial {t+1}/{TRIALS}...", end="", flush=True)
        start_time = time.time()
        
        model = build_paper_minimal_model(activation=act)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
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
        
    results[name] = trial_accuracies
    avg_acc = np.mean(trial_accuracies)
    max_acc = np.max(trial_accuracies)
    success_count = sum(1 for a in trial_accuracies if a > 0.99)
    print(f"  >> Avg: {avg_acc*100:.2f}%, Best: {max_acc*100:.2f}%, Successes: {success_count}/{TRIALS}")

with open('paper_minimal_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nExperiment Complete. Results saved to paper_minimal_results.json")
