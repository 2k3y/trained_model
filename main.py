import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# === НАСТРОЙКИ ===
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224) # Полное зрение для мелких деталей тракторного следа
BATCH_SIZE = 32
EPOCHS = 15 # Ставим с запасом, автоматика сама остановит, когда нужно

print(f"Используется устройство: {tf.test.gpu_device_name() if tf.test.is_gpu_available() else 'CPU'}")

# === 1. ЗАГРУЗКА ДАННЫХ ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === 2. СОЗДАНИЕ МОДЕЛИ ===
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomBrightness(0.3),
  layers.RandomContrast(0.3),
  layers.RandomZoom(0.2), # Защита от перепадов высоты
])

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False # Жестко замораживаем базу (Fine-tuning отменен)

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- КОЛЛБЭКИ ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

print("\n===== НАЧАЛО ОБУЧЕНИЯ =====")
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

# === 3. СОХРАНЕНИЕ ===
model_name = "agro_cnn_model.keras"
model.save(model_name)
print(f"\n✅ Нейросеть сохранена: {model_name}")
print("Теперь скопируй этот файл в папку с автопилотом!")