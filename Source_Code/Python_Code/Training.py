import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

gpus = tf.config.list_physical_devices('GPU')
print("[INFO] GPU devices:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

print("[INFO] TensorFlow:", tf.__version__)

DATASET_DIR = "/kaggle/input/wafer-defect-detection"
MODEL_SAVE_PATH = "/kaggle/working/mobilenetv2_lswmd_9class.keras"
PLOT_DIR = "/kaggle/working/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_WARMUP = 8
EPOCHS_FINE = 15
LR_WARMUP = 3e-4
LR_FINE = 1e-4
SEED = 42
NUM_CLASSES = 9

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("[INFO] Class names:", class_names)

train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)

labels = np.concatenate([y.numpy() for _, y in train_ds])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal")
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_WARMUP),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )
]

history_warmup = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_WARMUP,
    class_weight=class_weights,
    callbacks=callbacks
)

base_model.trainable = True
for layer in base_model.layers[:120]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(MODEL_SAVE_PATH)
print("[DONE] Model saved at:", MODEL_SAVE_PATH)

acc = history_warmup.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history_warmup.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history_warmup.history["loss"] + history_fine.history["loss"]
val_loss = history_warmup.history["val_loss"] + history_fine.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 – LSWMD Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MobileNetV2 – LSWMD Loss")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.close()

y_true, y_pred = [], []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – LSWMD (9 Classes)")
plt.close()

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_true, y_pred, target_names=class_names))
