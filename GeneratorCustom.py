import numpy as np
from tensorflow.keras.utils import Sequence
import cv2

class DataLoader(Sequence):
    def __init__(self, all_filenames, input_size = (256, 256), batch_size = 32, shuffle = True, seed = 123, encode: dict = None) -> None:
        super(DataLoader, self).__init__()
        assert encode != None,  'Not empty !'
        self.all_filenames = all_filenames
        self.input_size = input_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        np.random.seed(seed)
        self.on_epoch_end()
    def processing(self, mask):
        d  = [] 
        for i in mask.reshape(-1,3):
            d.append(self.encode[tuple(i)])
        return np.array(d).reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)

    def __len__(self):
        return int(np.floor(len(self.all_train) / self.batch_size))
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        all_filenames_temp = [self.all_filenames[k] for k in indexes]
        X, Y = self.__data_generation(all_filenames_temp)
        return X, Y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, all_filenames_temp):
        batch = len(all_filenames_temp)
        X = np.empty(shape=(batch, *self.input_size, 3))
        Y = np.empty(shape=(batch, *self.input_size, 1))
        for i, (fn, label_fn) in enumerate(all_filenames_temp):
            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.resize(img, self.input_size)
            img /= 255.
            mask = cv2.imread(label_fn)
            mask = cv2.resize(mask, self.input_size)
            mask = self.processing(mask)
            X[i,] = img
            Y[i,] = mask
        return X, Y

