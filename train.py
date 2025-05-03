"""
MIT License

Copyright (c) 2025 Seonghyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras import models, layers

# Load annotation CSV and encode class labels
csv_path = os.path.join("train", "_annotations.csv")
df = pd.read_csv(csv_path)
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['class'])
class_names = le.classes_
num_classes = len(class_names)
image_paths = df['filename'].unique()


# Load and preprocess image data, return images and corresponding bounding boxes

def load_data(df, img_dir=""):
    images = []
    targets = []

    for img_path in image_paths:
        full_path = os.path.join(img_dir, img_path)
        img = tf.io.read_file(full_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (416, 416)) / 255.0

        labels = df[df['filename'] == img_path]
        boxes = labels[['xmin', 'ymin', 'xmax', 'ymax']].values / 416.0 
        classes = labels['label_id'].values

        target = {
            "boxes": boxes.astype(np.float32),
            "classes": classes.astype(np.int32)
        }

        images.append(img)
        targets.append(target)

    return images, targets


# Build a simple CNN-based classification model 
def build_model(input_shape=(416, 416, 3), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    print("[*] Loading dataset...")
    img_dir = "/Users/sunghyunkim/Desktop/ReCloset.v3i.tensorflow/train"
    images, targets = load_data(df, img_dir)

    # For demo: extract classes only
    X = tf.stack(images)
    y = np.array([target['classes'][0] for target in targets])  # use first class per image

    print("[*] Building model...")
    model = build_model(num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("[*] Training model...")
    model.fit(X, y, epochs=5, batch_size=8)

    print("[*] Training complete. Saving model...")
    model.save("reclosetmodel.h5")


if __name__ == "__main__":
    main()
