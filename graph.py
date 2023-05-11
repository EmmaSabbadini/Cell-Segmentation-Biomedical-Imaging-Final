import os 
import time
import cv2
from tqdm import tqdm
import numpy as np
import skimage.draw
import random
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.io as sio
from PIL import Image
import scipy
import scipy.ndimage
import keras.backend as K
import segmentation_models as sm
from keras import utils

experiment_name = 'exp'

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot([0.9028, 0.8454, 0.8421, 0.7786],linewidth=4)
plt.plot([0.8887, 0.8419, 0.839, 0.7800],linewidth=4)
plt.title('{} loss'.format(experiment_name))
#plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)

# Plot training & validation iou_score values

plt.subplot(132)
plt.plot([0.1414, 0.1962, 0.2010, 0.2369],linewidth=4)
plt.plot([0.1444, 0.2, 0.2155, 0.2600],linewidth=4)
plt.title('{} IOU score'.format(experiment_name))
#plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values

plt.subplot(133)
plt.plot([0.1860, 0.2472, 0.2525, 0.3008],linewidth=4)
plt.plot([0.2048, 0.2784, 0.28, 0.33],linewidth=4)
plt.title('{} F1 score'.format(experiment_name))
#plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
plt.savefig('/Users/calliesardina/Spring_2023/CSCI3397/Cell-Segmentation-Biomedical-Imaging-Final/graph.png', dpi=300)
plt.show()