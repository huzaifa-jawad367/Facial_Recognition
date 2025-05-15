from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL
import cv2 as cv
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json

from Face_detect import detect_faces
from utils import pad_to_square_np



faces, runtime = detect_faces('Face Recognition Dataset-1/Dr. Adnan ul Hasan/9d13ac91-c3f1-446c-b7f2-8e10d4483195.jpg', 'mtcnn')
face_arr = faces[0]['face']
print(face_arr.shape)

square_face = pad_to_square_np(face_arr)

reshaped_face = square_face.reshape((160,160,3))
print(reshaped_face.shape)

plt.imshow(square_face)
plt.axis('off')
# plt.title(f"mtcnn: {len(faces)} faces in {runtime:.2f}s")

# Save to file instead of (or in addition to) showing on screen
plt.savefig(
    'detected_face.png',         # output filename
    bbox_inches='tight',         # trim whitespace
    pad_inches=0,                # no extra padding
    dpi=150                      # adjust resolution if you like
)

plt.close()  # free the figure


json_file = open('Assets/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# model = model_from_json(loaded_model_json)
# model.load_weights('Assets/facenet_keras.h5')

# FRmodel = model