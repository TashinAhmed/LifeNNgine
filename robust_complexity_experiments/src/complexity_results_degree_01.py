
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def life_step(grid, rule_b, rule_s):
    """
    Generic Life-like cellular automaton step.
    rule_b: list of neighbor counts for birth
    rule_s: list of neighbor counts for survival
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

class PolynomialActivation(layers.Layer):
    """
    Learnable polynomial activation function of degree d.
    f(x) = sum(w_i * x^i) for i in [0, d]
    """
    def __init__(self, degree=2, **kwargs):
        super(PolynomialActivation, self).__init__(**kwargs)
        self.degree = degree

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.coeffs = []
        for i in range(self.degree + 1):
            # Initialize w1=1, others=0 for identity-like start, or random
            # Here we initialize to small random values to break symmetry, 
            # except w1=1 to encourage gradient flow
            if i == 1:
                initializer = 'ones'
            else:
                initializer = 'zeros' # 'glorot_uniform' might be better for high degrees?
                # Let's stick to zeros/ones for stability and let it learn deviations
            
            w = self.add_weight(name=f'w{i}', shape=(1, 1, 1, num_channels), 
                                initializer=initializer, trainable=True)
            self.coeffs.append(w)
        
    def call(self, inputs):
        output = self.coeffs[0] # w0 * x^0
        x_pow = inputs
        output += self.coeffs[1] * x_pow 
        
        for i in range(2, self.degree + 1):
            x_pow = x_pow * inputs # x^i
            output += self.coeffs[i] * x_pow
            
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"degree": self.degree})
        return config

def build_poly_kan(grid_size, degree=2, filters=2):
    """
    Builds a CNN with Polynomial activations (PolyKAN).
    Structure: Conv -> Poly -> Conv -> Poly -> Output
    """
    inputs = layers.Input(shape=(grid_size, grid_size, 1))
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = PolynomialActivation(degree=degree, name='poly1')(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = PolynomialActivation(degree=degree, name='poly2')(x)
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

def run_experiments():
    GRID_SIZE = 20
    NUM_TRAIN = 5000
    NUM_VAL = 1000
    rules = {
        "Game of Life": {"b": [3], "s": [2, 3]},
        "Replicator":   {"b": [1,3,5,7], "s": [1,3,5,7]},
        "Fredkin":      {"b": [1,3,5,7], "s": [0,2,4,6,8]}
    }
    degrees = [2, 4, 8, 16]
    results = {}
    for rule_name, rule_def in rules.items():
        print(f"\n=== Testing Rule: {rule_name} (B{rule_def['b']}/S{rule_def['s']}) ===")
        results[rule_name] = {}
        print("Generating data...")
        X_train, y_train = generate_data(NUM_TRAIN, GRID_SIZE, rule_def['b'], rule_def['s'])
        X_val, y_val = generate_data(NUM_VAL, GRID_SIZE, rule_def['b'], rule_def['s'])
        
        for deg in degrees:
            print(f"  Testing Degree {deg}...")
            model = build_poly_kan(GRID_SIZE, degree=deg, filters=4)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
            history = model.fit(X_train, y_train, epochs=40, batch_size=64, verbose=0)
            acc = evaluate_exact_match(model, X_val, y_val)
            print(f"    -> Accuracy: {acc*100:.2f}%")
            results[rule_name][f"degree_{deg}"] = float(acc)
            
    with open('complexity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n=== Final Results ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_experiments()
