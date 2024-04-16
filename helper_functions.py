import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from PIL import Image, ImageDraw
from helper_functions import *
import os
import datetime
import tensorflow as tf

###########################################################################################
def model_checkpoint(model, dir):
    # Generate a unique checkpoint path based on current timestamp
    checkpoint_name = f'{model.name}'
    checkpoint_path = os.path.join(dir, f"cp_{checkpoint_name}.ckpt")
    
    # Define the ModelCheckpoint callback with the unique checkpoint path
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        save_weights_only=True,
        verbose=0
    )
    return model_checkpoint

###########################################################################################

def plot_loss_regression(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

###########################################################################################

def plot_loss_classifier(history):
  """
  Returns separate loss curves for training and validation metrics for classification
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()

############################################################################################

def plot_bounding_boxes(images, labels_df, bbox_preds, figsize=(15, 18)):
    num_images = len(images[:15])
    
    plt.figure(figsize=figsize)
    
    for i in range(num_images):
        image = images[i]
        plt.subplot(5, 3, i+1)
        plt.imshow(image)
        
        # Retrieve actual bounding box coordinates
        x_min_actual = labels_df['x_min'][i]
        y_min_actual = labels_df['y_min'][i]
        x_max_actual = labels_df['x_max'][i]
        y_max_actual = labels_df['y_max'][i]
        
        # Calculate width and height of the actual bounding box
        width_actual = x_max_actual - x_min_actual
        height_actual = y_max_actual - y_min_actual
        
        # Draw actual bounding box in red
        bbox_actual = patches.Rectangle((x_min_actual, y_min_actual), width_actual, height_actual,
                                         linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(bbox_actual)
        
        # Retrieve predicted bounding box coordinates
        x_min_pred, y_min_pred, x_max_pred, y_max_pred = bbox_preds[i]
        
        # Calculate width and height of the predicted bounding box
        width_pred = x_max_pred - x_min_pred
        height_pred = y_max_pred - y_min_pred
        
        # Draw predicted bounding box in green
        bbox_pred = patches.Rectangle((x_min_pred, y_min_pred), width_pred, height_pred,
                                      linewidth=2, edgecolor='g', facecolor='none')
        plt.gca().add_patch(bbox_pred)
        
        plt.title(f"Image {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

############################################################################################

def create_model(input_shape, num_classes, name):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    vehicle_class = layers.Dense(num_classes, activation='softmax', name='vehicle_class')(x)
    bounding_box = layers.Dense(4, name='bounding_box')(x)

    model = keras.Model(inputs=inputs, outputs=[vehicle_class, bounding_box], name=name)
    return model

############################################################################################

def create_efficientnet_model(input_shape, num_classes, backbone, name):
    inputs = keras.Input(shape=input_shape)
    base_model = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(base_model)
    y = layers.Dense(64, activation='relu')(x)
    vehicle_class = layers.Dense(num_classes, activation='softmax', name='vehicle_class')(x)
    bounding_box = layers.Dense(4, name='bounding_box')(y)

    model = keras.Model(inputs=inputs, outputs=[vehicle_class, bounding_box], name=name)
    return model