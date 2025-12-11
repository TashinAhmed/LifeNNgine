
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def life_step(grid, rule_b, rule_s):
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
    def __init__(self, degree=2, **kwargs):
        super(PolynomialActivation, self).__init__(**kwargs)
        self.degree = degree

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.coeffs = []
        for i in range(self.degree + 1):
            if i == 1:
                initializer = 'ones'
            else:
                initializer = 'zeros' # 'glorot_uniform' might help for deep nets?
            w = self.add_weight(name=f'w{i}', shape=(1, 1, 1, num_channels), 
                                initializer=initializer, trainable=True)
            self.coeffs.append(w)
        
    def call(self, inputs):
        output = self.coeffs[0]
        x_pow = inputs
        output += self.coeffs[1] * x_pow 
        for i in range(2, self.degree + 1):
            x_pow = x_pow * inputs
            output += self.coeffs[i] * x_pow
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"degree": self.degree})
        return config

def build_flexible_poly_kan(grid_size, degree=2, filters=4, num_layers=2):
    """
    Builds a deeper/wider PolyKAN.
    num_layers: number of [Conv -> Poly] blocks before the final output.
    """
    inputs = layers.Input(shape=(grid_size, grid_size, 1))
    x = inputs
    
    for i in range(num_layers):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = PolynomialActivation(degree=degree, name=f'poly_{i}')(x)
        
    # Final Output Layer
    # Just a linear projection to 1 channel + sigmoid? 
    # Or another Conv(1)->Poly block then sigmoid?
    # Let's do a simple projection.
    
    # NOTE: Previous working model had a bottleneck Conv(1) layer before output.
    # To strictly scale width, we should NOT have 1-channel bottlenecks in the middle.
    
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_exact_match(model, X, y):
    preds = model.predict(X, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y, axis=(1, 2, 3))
    return np.mean(matches)

def run_experiments_02():
    GRID_SIZE = 20
    NUM_TRAIN = 5000
    NUM_VAL = 1000
    
    # Fredkin Rule
    rule_b = [1,3,5,7]
    rule_s = [0,2,4,6,8]
    print(f"\n=== Generating Data for Fredkin Rule ===")
    X_train, y_train = generate_data(NUM_TRAIN, GRID_SIZE, rule_b, rule_s)
    X_val, y_val = generate_data(NUM_VAL, GRID_SIZE, rule_b, rule_s)
    
    # Search Space
    widths = [4, 8, 16, 32]
    depths = [2, 3, 4]
    degrees = [2, 4] 
    
    results = {}
    
    print("\n=== Starting Depth/Width/Degree Sweep ===")
    
    for deg in degrees:
        for d in depths:
            for w in widths:
                config_name = f"D{d}_W{w}_Deg{deg}"
                print(f"Testing {config_name}...")
                
                model = build_flexible_poly_kan(GRID_SIZE, degree=deg, filters=w, num_layers=d)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                              loss='binary_crossentropy', 
                              metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=40, batch_size=64, verbose=0)
                
                acc = evaluate_exact_match(model, X_val, y_val)
                print(f"  -> Accuracy: {acc*100:.2f}%")
                
                results[config_name] = float(acc)
                
                if acc > 0.99:
                    print(f"!!! SOLVED Fredkin with {config_name} !!!")
    
    with open('complexity_results_02.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiments_02()
