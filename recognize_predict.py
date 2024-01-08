import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tqdm import tqdm
import sys
import config 
import warnings
warnings.filterwarnings('ignore')

sys.path.append(config.DEEP_DIR)
from deepface import DeepFace


class RecognizeModel():
  
  def __init__(self, datasets, weight_dir, model_name):
    self.weight_dir = weight_dir 
    self.datasets = datasets
    self.models = self.load_model()
    self.model_name = model_name
    self.dirs = self.get_dir()

  def load_model(self):
    return tf.keras.models.load_model(self.weight_dir)
  
  def extract_features(self, image):
    embedding_objs = DeepFace.represent(image, model_name=self.model_name, enforce_detection=False)
    return embedding_objs[0]['embedding']
  
  def get_dir(self):
      return os.listdir(self.datasets)
  
  def recognize_predict(self, image):
    dirs = self.dirs 

    feature_vector = self.extract_features(image)

    prediction = self.models.predict([feature_vector])

    predicted_class = np.argmax(prediction)

    confidence = prediction[0][predicted_class] * 100 


    result = {'confidence':confidence , 'name': dirs[np.argmax(prediction)]}
    return result


