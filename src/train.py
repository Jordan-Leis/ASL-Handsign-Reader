import argparse
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ASL classifier")
    parser.add_argument("--data-dir", required=True, help="Path to ASL dataset")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--model-path",
        default="models/asl_model.h5",
        help="Where to save the trained model",
    )
    args = parser.parse_args()

    datagen = ImageDataGenerator(validation_split=0.1)
    train_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=(64, 64),
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=(64, 64),
        subset="validation",
    )

    model = build_model(num_classes=train_gen.num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

    # Save class names for inference
    class_file = os.path.splitext(args.model_path)[0] + "_classes.txt"
    with open(class_file, "w") as f:
        for cls in train_gen.class_indices:
            f.write(f"{cls}\n")
    print(f"Class names saved to {class_file}")


if __name__ == "__main__":
    main()
