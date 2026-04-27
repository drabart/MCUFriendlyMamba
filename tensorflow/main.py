import tensorflow as tf
from lite_mamba import TFPTCNMamba

# 1. Load and Preprocess Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape to (batch, sequence_length, d_model)
# 28 rows (sequence), 28 columns (features)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build the Model
# We replace the standard Dense/Flatten layers with the Mamba block
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    
    # Mamba block expects d_model=28 (the width of the image)
    TFPTCNMamba(d_model=28, d_conv=3, conv_dilations=(1, 2, 4, 8)),
    
    # Global Pooling to reduce the sequence dimension before the classifier
    tf.keras.layers.GlobalAveragePooling1D(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# 3. Training setup
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 4. Fit
print("Starting training on 7800 XT...")
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 5. Evaluate
model.evaluate(x_test, y_test, verbose=2)
