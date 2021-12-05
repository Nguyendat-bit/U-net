import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from GeneratorCustom import DataGenerator
import numpy as np
import sklearn
import cv2
import tensorflow as tf

def encode_label(mask):
    # input (batch, rows, cols, channels)
    label = [] 
    for i in mask.reshape(-1,3):
        label.append(tuple(i))
    label = set(label)
    encoder = dict((j,i) for i,j in enumerate(label)) # key is tuple 
    with open('label.pickle', 'wb') as handel:
        pickle.dump(encoder, handel, protocol= pickle.HIGHEST_PROTOCOL)
    return encoder
def decode_label(predict, label):
    predict = tf.squeeze(predict, axis = 0)
    predict = np.argmax(predict, axis = 2) 
    d = list(map( lambda x: label[int(x)], predict.reshape(-1,1)))
    img =  np.array(d).reshape(predict.shape[0], predict.shape[1], 3)
    return img
def DataLoader(all_train_filename, all_mask,  all_valid_filename = None, input_size = (256,256), batch_size = 4, shuffle = True, seed = 123, color_mode = 'hsv') -> None:
    mask_folder = sklearn.utils.shuffle(all_mask, random_state = 47)[:8]
    mask = [tf.image.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), input_size, method = 'nearest') for img in mask_folder ]
    mask = np.array(mask)
    encode = encode_label(mask)
    train = DataGenerator(all_train_filename, input_size, batch_size , shuffle, seed, encode, color_mode)
    if all_valid_filename == None: 
        return train, None
    else:
        valid = DataGenerator(all_valid_filename, input_size, batch_size, shuffle, seed, encode, color_mode)
        return train, valid