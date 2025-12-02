import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def life_step(grid):
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
        next_state = life_step(state)
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
    return models.Model(inputs=inputs, outputs=outputs)

GRID_SIZE = 20
print("Generating Data...")
X_train, y_train = generate_data(10000, GRID_SIZE) # Increased to 10k for reliability
X_val, y_val = generate_data(1000, GRID_SIZE)

print("Training QuadKAN Model...")
model = None
for attempt in range(5):
    print(f"Attempt {attempt+1}...")
    model = build_kan_model(GRID_SIZE, filters=2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

    # Verify acc
    preds = model.predict(X_val, verbose=0)
    preds_binary = (preds > 0.5).astype(np.float32)
    matches = np.all(preds_binary == y_val, axis=(1, 2, 3))
    acc = np.mean(matches)
    print(f"Model Accuracy: {acc*100:.2f}%")
    
    if acc > 0.99:
        break
else:
    print("Failed to train a good model after 5 attempts. Animation will be poor.")

if acc < 0.9:
    print("Warning: Model accuracy is low. Animation might show incorrect behavior.")

print("Generating Animation Sequence...")

steps = 30
initial_state = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2]).astype(np.float32)
initial_state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
initial_state[1, 2] = 1
initial_state[2, 3] = 1
initial_state[3, 1] = 1
initial_state[3, 2] = 1
initial_state[3, 3] = 1

current_state = initial_state
history = []

layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

for _ in range(steps):
    activations = activation_model.predict(current_state[np.newaxis, ..., np.newaxis], verbose=0)
    frame_data = {
        'input': activations[0][0, :, :, 0],
        'conv1': activations[1][0, :, :, :], # 2 filters
        'quad1': activations[2][0, :, :, :], # 2 filters
        'conv2': activations[3][0, :, :, 0],
        'quad2': activations[4][0, :, :, 0],
        'output': activations[5][0, :, :, 0]
    }
    history.append(frame_data)
    current_state = (activations[5][0, :, :, 0] > 0.5).astype(np.float32)

fig = plt.figure(figsize=(14, 8))
plt.style.use('dark_background')

# Layout:
# Input | Conv1_F1 | Quad1_F1 | Conv2 | Output
#       | Conv1_F2 | Quad1_F2 | Quad2 | 

gs = fig.add_gridspec(2, 5)

ax_input = fig.add_subplot(gs[:, 0])
ax_input.set_title("Input (t)")

ax_c1_f1 = fig.add_subplot(gs[0, 1])
ax_c1_f1.set_title("L1 Conv F1")
ax_c1_f2 = fig.add_subplot(gs[1, 1])
ax_c1_f2.set_title("L1 Conv F2")

ax_q1_f1 = fig.add_subplot(gs[0, 2])
ax_q1_f1.set_title("L1 Quad F1")
ax_q1_f2 = fig.add_subplot(gs[1, 2])
ax_q1_f2.set_title("L1 Quad F2")

ax_c2 = fig.add_subplot(gs[0, 3])
ax_c2.set_title("L2 Conv")
ax_q2 = fig.add_subplot(gs[1, 3])
ax_q2.set_title("L2 Quad")

ax_out = fig.add_subplot(gs[:, 4])
ax_out.set_title("Output (t+1)")

axes = [ax_input, ax_c1_f1, ax_c1_f2, ax_q1_f1, ax_q1_f2, ax_c2, ax_q2, ax_out]
images = []

for ax in axes:
    ax.axis('off')
    img = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='viridis', vmin=0, vmax=1)
    images.append(img)

# Adjust vmin/vmax for intermediate layers dynamically or set fixed range
# For simplicity, we let matplotlib normalize, or we can fix it if we know ranges.
# Let's use set_data in update loop.

def update(frame_idx):
    data = history[frame_idx]
    
    images[0].set_data(data['input'])
    images[0].set_clim(0, 1)
    
    images[1].set_data(data['conv1'][:, :, 0])
    images[1].autoscale()
    images[2].set_data(data['conv1'][:, :, 1])
    images[2].autoscale()
    
    images[3].set_data(data['quad1'][:, :, 0])
    images[3].autoscale()
    images[4].set_data(data['quad1'][:, :, 1])
    images[4].autoscale()
    
    images[5].set_data(data['conv2'])
    images[5].autoscale()
    
    images[6].set_data(data['quad2'])
    images[6].autoscale()
    
    images[7].set_data(data['output'])
    images[7].set_clim(0, 1)
    
    fig.suptitle(f"QuadKAN Network Dynamics - Step {frame_idx+1}/{steps}", fontsize=16)
    return images

ani = animation.FuncAnimation(fig, update, frames=steps, interval=200, blit=False)

output_file = 'kan_dynamics.gif'
print(f"Saving animation to {output_file}...")
ani.save(output_file, writer='pillow', fps=5)
print("Done.")
