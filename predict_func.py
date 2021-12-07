import cv2
import tensorflow as tf
import data
import numpy as np
def predict(model, image_test, label, color_mode, size):
    image = cv2.imread(image_test)
    if color_mode == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, size, method= 'nearest')
    image = tf.cast(image, tf.float32) 
    image_norm = image / 255.
    image_norm = tf.expand_dims(image_norm, axis= 0)
    new_image = model(image_norm)
    image_decode = data.decode_label(new_image, label)
    predict_img = image * 0.5 + image_decode * 0.5
    return np.floor(predict_img)[0].astype('int'), new_image