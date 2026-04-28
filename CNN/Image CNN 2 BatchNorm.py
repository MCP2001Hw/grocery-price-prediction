import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from PIL import Image

print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# (1) LOAD DATA
# ============================================================================

df = pd.read_csv("Data.csv")
urls = df["Image link of the product"].values
prices = df["Price of the product"].values

# ============================================================================
# (2) IMAGE PREPROCESSING
# ============================================================================

# Use a thread-local requests.Session for faster HTTP I/O when using threads
_thread_local = threading.local()

def _get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def load_and_preprocess_image(url, target_size=(128, 128)):
    """Download image from URL using a thread-local Session and normalize to [0, 1].

    Returns None on failure so downstream code can filter bad downloads.
    """
    try:
        sess = _get_session()
        response = sess.get(url, timeout=6)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(target_size)
        return np.array(img) / 255.0
    except Exception:
        return None

# ============================================================================
# (3) LOAD OR CACHE DATASET
# ============================================================================

import os

# Resolve paths relative to script for .npy cache files
script_dir = Path(__file__).parent

x_images_path = script_dir / "x_images.npy"
y_prices_path = script_dir / "y_prices.npy"
names_path = script_dir / "names.npy"

subset = 4000  # Increase once code runs smoothly

# Check for cached files
cached = all([x_images_path.exists(), y_prices_path.exists(), names_path.exists()])

if not cached:
    print("Downloading and preprocessing images (parallel)...")
    n_to_download = min(subset, len(urls))
    # choose a reasonable number of worker threads for I/O-bound downloads
    max_workers = min(32, (os.cpu_count() or 1) * 5, n_to_download)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # executor.map preserves order; wrap with tqdm for progress
        images = list(tqdm(ex.map(load_and_preprocess_image, urls[:n_to_download]), total=n_to_download, desc="Downloading"))
    valid_idxs = [i for i, img in enumerate(images) if img is not None]
    x = np.array([img for img in images if img is not None])
    y = np.array(prices[:len(x)])
    names = np.array(df["Name of the product"].values[:subset])[valid_idxs]

    print("Saving to disk...")
    np.save(str(x_images_path), x)
    np.save(str(y_prices_path), y)
    np.save(str(names_path), names, allow_pickle=True)
    print("Cached successfully")
else:
    print("Loading cached dataset...")
    x = np.load(str(x_images_path), mmap_mode='r')
    y = np.load(str(y_prices_path), mmap_mode='r')
    names = np.load(str(names_path), allow_pickle=True)

# ============================================================================
# (4) SPLIT DATASET
# ============================================================================

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]
names_train, names_test = names[:split], names[split:]

# Create tf.data pipelines for efficient batching
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# ============================================================================
# (5) BUILD CNN MODEL
# ============================================================================

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='linear')  # Regression: continuous price output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# ============================================================================
# (6) TRAIN MODEL WITH EARLY STOPPING
# ============================================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    verbose=1,
    callbacks=[early_stopping]
)

loss, mae = model.evaluate(val_ds)
# Compute RMSE from predictions (most direct/accurate across the whole test set)
preds = model.predict(x_test, verbose=0).flatten()
rmse_preds = np.sqrt(np.mean((preds - y_test) ** 2))
print(f"\nFinal Test MAE: £{mae:.2f}")
print(f"Final Test RMSE (predictions-based): £{rmse_preds:.2f}")

# ============================================================================
# (7) VISUALIZE TRAINING PROGRESS
# ============================================================================

plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Absolute Error (£)', fontsize=12)
plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# (8) SAMPLE PREDICTIONS
# ============================================================================

# Show 5 random test samples
random_indices = np.random.choice(len(x_test), 5, replace=False)
predicted_prices = model.predict(x_test[random_indices], verbose=0)
print("\n" + "="*60)
print("Sample Predictions (Random):")
print("="*60)
for i, idx in enumerate(random_indices):
    print(f"Product: {names_test[idx]}")
    print(f"  Predicted: £{predicted_prices[i, 0]:.2f}")
    print(f"  Actual:    £{y_test[idx]:.2f}")
    print()

# ============================================================================
# (9) VISUALIZE LAYERS
# ============================================================================

# Visualize the learned feature maps from convolutional layers
# Reuse the same random indices used for sample predictions so visuals match the samples
vis_indices = list(random_indices[:3]) if 'random_indices' in globals() else [0, 1, 2]
# ensure indices are valid
vis_indices = [int(i) for i in vis_indices]
vis_indices = [i if 0 <= i < len(x_test) else 0 for i in vis_indices]

# Create model that outputs layer activations (only convolutional layers)
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name.lower()]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

# Predict on the selected test images to get layer activations
activations = activation_model.predict(x_test[vis_indices], verbose=0)

num_images = len(vis_indices)
num_layers = min(3, len(activations))
fig, axes = plt.subplots(num_images, num_layers, figsize=(4 * num_layers, 3 * num_images))
if num_images == 1 and num_layers == 1:
    axes = np.array([[axes]])
elif num_images == 1:
    axes = np.expand_dims(axes, 0)
elif num_layers == 1:
    axes = np.expand_dims(axes, 1)

for layer_idx in range(num_layers):
    for img_idx in range(num_images):
        # activations[layer_idx] shape: (num_images, H, W, C)
        feature_maps = activations[layer_idx][img_idx]  # H x W x C
        # Choose the channel with the largest variance (most informative) to visualise
        try:
            channel_variances = np.var(feature_maps, axis=(0, 1))  # length C
            best_channel = int(np.argmax(channel_variances))
        except Exception:
            best_channel = 0

        fmap = feature_maps[:, :, best_channel]
        fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        ax = axes[img_idx, layer_idx]
        ax.imshow(fmap_norm, cmap='viridis')
        ax.grid(False)
        if img_idx == 0:
            ax.set_title(f"Conv Layer {layer_idx + 1}\n(channel {best_channel})")
        if layer_idx == 0:
            ax.set_ylabel(f"Img {vis_indices[img_idx]}")

plt.tight_layout()
plt.show()
