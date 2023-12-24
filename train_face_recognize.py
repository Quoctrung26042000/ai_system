import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import sys
sys.path.append('./deep_face')
from deepface import DeepFace

# Function to extract features from images using DeepFace
def extract_features(img_path, model_name='Facenet512'):
    embedding_objs = DeepFace.represent(img_path, model_name=model_name, enforce_detection=False)
    return embedding_objs[0]['embedding']

def main(data_dir, model_name):
    # List all directories (persons)
    dirs = os.listdir(data_dir)
    data = []
    for dir in tqdm(dirs):
        for img in os.listdir(os.path.join(data_dir, dir)):
            features = extract_features(os.path.join(data_dir, dir, img))
            features.insert(0, dir)
            data.append(features)


    column_names = ['person name']
    for i in range(512):
        column_names.append(f'f{i+1}')

    df_org = pd.DataFrame(data, columns=column_names)
    df_org.to_csv('face_features.csv', index=False)

    df = df_org.replace(dirs, [i for i in range(len(dirs))])
    df = df.sample(frac=1)

    x_train = df.iloc[:, 1:]
    y_train = df['person name']

    # Model creation
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, activation='tanh', input_shape=(512,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(dirs), activation='softmax')
    ])
    model.compile(tf.keras.optimizers.Adamax(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Model training
    output = model.fit(x_train, y_train, epochs=60)

    # Save the trained model
    model.save('weights/model.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the facial recognition model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--model_name", type=str, default='Facenet512', help="Name of the feature extraction model")

    args = parser.parse_args()

    data_directory = args.data_dir
    selected_model_name = args.model_name

    main(data_directory, selected_model_name)
