import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

model = load_model('Cancer_detection_model.h5')

data_cat = ['all_benign', 'all_early', 'all_pre', 'all_pro', 'lymph_cll', 'lymph_fl', 'lymph_mcl']

image = "C:/Users/sonia/Downloads/Sap_013 (1).jpg"
image = tf.keras.utils.load_img(image, target_size=(180,180))
img_arr = tf.keras.utils.img_to_array(image)
#img_arr /= 255.0 
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)
score = tf.nn.softmax(predict)

print('Cancer image belongs to {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))