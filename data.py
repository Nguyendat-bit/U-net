import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from GeneratorCustom import DataGenerator
import numpy as np
import sklearn
import cv2
import tensorflow as tf
from sklearn.cluster import KMeans

def encode_label(mask):
    # input (batch, rows, cols, channels)
    colors = np.unique(mask.reshape(-1,3), axis = 0)
    encoder = dict((tuple(j),i) for i,j in enumerate(colors)) # key is tuple 
    _label = dict((j, list(i)) for i,j in encoder.items())
    with open('label.pickle', 'wb') as handel:
        pickle.dump(_label, handel, protocol= pickle.HIGHEST_PROTOCOL)
    return encoder
def encode_label_with_Kmeans(mask, classes):
    kmean = KMeans(classes, max_iter= 400)
    kmean.fit(mask)
    pred = kmean.predict(mask)
    classes_real =  len(set(pred))
    print(f'classes: {classes_real}')
    label = dict((j, i.tolist()) for i,j in list(zip(mask, pred))) # key is tuple 
    with open('label.pickle', 'wb') as handel:
        pickle.dump(label, handel, protocol= pickle.HIGHEST_PROTOCOL)
    with open('kmean.pickle', 'wb') as handle:
        pickle.dump(kmean, handle, protocol= pickle.HIGHEST_PROTOCOL)
    return kmean

def decode_label(predict, label):
    predict = tf.squeeze(predict, axis = 0)
    predict = np.argmax(predict, axis = 2) 
    d = list(map( lambda x: label[int(x)], predict.reshape(-1,1)))
    img =  np.array(d).reshape(predict.shape[0], predict.shape[1], 3)
    return img
def DataLoader(all_train_filename, all_mask,  all_valid_filename = None, input_size = (256,256), batch_size = 4, shuffle = True, seed = 123, color_mode = 'hsv', function = None, encode_with_kmeans = False, classes = 0) -> None:
    mask_folder = sklearn.utils.shuffle(all_mask, random_state = 47)[:16]
    mask = [tf.image.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), (128, 128), method = 'nearest') for img in mask_folder ]
    mask = np.array(mask)
    kmean = None
    encode = None
    if function and encode_with_kmeans == False:
        mask = function(mask)
    if encode_with_kmeans == False:
        encode = encode_label(mask)
    elif encode_with_kmeans == True:
        kmean = encode_label_with_Kmeans(mask.reshape(-1,3), classes)
    train = DataGenerator(all_train_filename, input_size, batch_size , shuffle, seed, encode, kmean, color_mode, function)
    if all_valid_filename == None: 
        return train, None
    else:
        valid = DataGenerator(all_valid_filename, input_size, batch_size, shuffle, seed, encode, kmean, color_mode, function)
        return train, valid