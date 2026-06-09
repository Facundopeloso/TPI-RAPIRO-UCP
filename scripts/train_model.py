"""
scripts/train_model.py
Entrenamiento del clasificador de estados del estudiante.

Arquitectura: MobileNetV2 (Transfer Learning) + cabeza de clasificación custom
Cuantización: INT8 post-training quantization para despliegue en Raspberry Pi 4
Salida: models/mobilenetv2_int8.tflite

Dataset esperado:
    dataset/
        class_0/   # Estudiando
        class_1/   # Usando celular
        class_2/   # Puesto vacío

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 20 --batch 32
    python scripts/train_model.py --fine-tune         # activa fine-tuning fase 2
"""

import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_DIR = os.path.join(ROOT, "dataset")
MODELS_DIR  = os.path.join(ROOT, "models")
OUTPUT_PATH = os.path.join(MODELS_DIR, "mobilenetv2_int8.tflite")
SAVED_MODEL = os.path.join(MODELS_DIR, "mobilenetv2_saved")
PLOTS_DIR   = os.path.join(ROOT, "models", "plots")

IMG_SIZE    = (224, 224)
NUM_CLASSES = 3
CLASS_NAMES = ["Estudiando", "Usando celular", "Puesto vacio"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(batch_size: int, val_split: float = 0.2):
    """
    Carga imágenes desde dataset/class_X/ con augmentación.

    Arquitectura del pipeline:
        ImageDataGenerator -> augmentación en CPU durante entrenamiento
        No se guardan imágenes aumentadas en disco.
    """
    import tensorflow as tf

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    return train_gen, val_gen


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(fine_tune_layers: int = 0):
    """
    Construye MobileNetV2 con cabeza de clasificación custom.

    Arquitectura completa:
        Input (224, 224, 3)
            |
        MobileNetV2 base (ImageNet weights, frozen en fase 1)
            |   1280 feature maps @ 7x7
        GlobalAveragePooling2D
            |   vector 1280
        Dense(256, activation='relu')
            |
        Dropout(0.4)
            |
        Dense(NUM_CLASSES, activation='softmax')
            |
        Output (3,)  [P(estudiando), P(celular), P(ausente)]

    Parámetros totales:    ~3.4M
    Parámetros entrenables (fase 1): ~330K (solo cabeza)
    Parámetros entrenables (fase 2): varía según fine_tune_layers
    """
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # Fine-tuning fase 2: descongelar últimas N capas del base
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True
        print(f"Fine-tuning: {fine_tune_layers} capas del base descongeladas.")

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(epochs: int, batch_size: int, do_fine_tune: bool):
    import tensorflow as tf

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  RAPIRO — Entrenamiento MobileNetV2")
    print("=" * 60)
    print(f"  Dataset   : {DATASET_DIR}")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch     : {batch_size}")
    print(f"  Fine-tune : {do_fine_tune}")
    print(f"  Salida    : {OUTPUT_PATH}")
    print("=" * 60)

    train_gen, val_gen = load_dataset(batch_size)
    print(f"\nClases detectadas: {train_gen.class_indices}")
    print(f"Train: {train_gen.samples} imgs | Val: {val_gen.samples} imgs\n")

    # ---- Fase 1: entrenar solo la cabeza ----
    model = build_model(fine_tune_layers=0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SAVED_MODEL + "_best.h5", save_best_only=True,
            monitor="val_accuracy", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]

    print("\n--- Fase 1: entrenando cabeza (base frozen) ---")
    history1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=callbacks,
    )

    # ---- Fase 2: fine-tuning (opcional) ----
    history2 = None
    if do_fine_tune:
        print("\n--- Fase 2: fine-tuning ultimas 30 capas del base ---")
        model = build_model(fine_tune_layers=30)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),   # LR bajo para no destruir features
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history2 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=epochs // 2, callbacks=callbacks,
        )

    # ---- Evaluación final ----
    print("\n--- Evaluación en validación ---")
    loss, acc = model.evaluate(val_gen)
    print(f"  Val accuracy : {acc*100:.2f}%")
    print(f"  Val loss     : {loss:.4f}")

    # ---- Guardar SavedModel ----
    model.save(SAVED_MODEL)
    print(f"\nSavedModel guardado en: {SAVED_MODEL}")

    # ---- Convertir a TFLite INT8 ----
    convert_to_tflite(model, train_gen)

    # ---- Guardar gráficas ----
    save_plots(history1, history2)

    return acc


# ---------------------------------------------------------------------------
# Cuantización TFLite INT8
# ---------------------------------------------------------------------------

def convert_to_tflite(model, train_gen):
    """
    Convierte el modelo a TFLite con cuantización INT8 completa.

    Por qué INT8:
        - Reduce tamaño del modelo ~4x (float32 → int8)
        - Acelera inferencia en ARM Cortex-A72 (Raspberry Pi 4) ~2-3x
        - Permite uso del delegado XNNPACK para operaciones SIMD
        - Latencia objetivo: <500ms en Raspberry Pi 4

    Proceso de cuantización:
        1. Se toma un conjunto representativo de imágenes del train set
        2. El cuantizador calibra los rangos de activación (min/max) por capa
        3. Pesos y activaciones se mapean a int8 usando los rangos calibrados
        4. La precisión cae típicamente <1% respecto al modelo float32
    """
    import tensorflow as tf

    print("\n--- Convirtiendo a TFLite INT8 ---")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Dataset representativo para calibración INT8
    def representative_data_gen():
        count = 0
        for batch_images, _ in train_gen:
            for img in batch_images:
                yield [np.expand_dims(img.astype(np.float32), axis=0)]
                count += 1
                if count >= 200:
                    return

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  Modelo INT8 guardado: {OUTPUT_PATH}")
    print(f"  Tamaño: {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def save_plots(history1, history2=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado — omitiendo gráficas.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    acc  = history1.history["accuracy"]
    val  = history1.history["val_accuracy"]
    loss = history1.history["loss"]
    vloss= history1.history["val_loss"]

    if history2:
        acc  += history2.history["accuracy"]
        val  += history2.history["val_accuracy"]
        loss += history2.history["loss"]
        vloss+= history2.history["val_loss"]
        axes[0].axvline(len(history1.history["accuracy"]), color="gray",
                        linestyle="--", label="inicio fine-tune")
        axes[1].axvline(len(history1.history["loss"]), color="gray", linestyle="--")

    axes[0].plot(acc, label="train")
    axes[0].plot(val, label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(loss, label="train")
    axes[1].plot(vloss, label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(out, dpi=120)
    print(f"  Curvas guardadas en: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar clasificador RAPIRO")
    parser.add_argument("--epochs",    type=int,  default=15,    help="Epochs fase 1 (default: 15)")
    parser.add_argument("--batch",     type=int,  default=16,    help="Batch size (default: 16)")
    parser.add_argument("--fine-tune", action="store_true",      help="Activar fine-tuning fase 2")
    args = parser.parse_args()

    accuracy = train(
        epochs=args.epochs,
        batch_size=args.batch,
        do_fine_tune=args.fine_tune,
    )

    print("\n" + "=" * 60)
    print(f"  Entrenamiento finalizado. Val accuracy: {accuracy*100:.2f}%")
    print(f"  Modelo listo en: {OUTPUT_PATH}")
    print("=" * 60)
