import matplotlib.pyplot as plt 
import cv2
import predict_func
import tensorflow as tf
from data import *
import numpy as np
def show_history(history, validation : bool = False):
    if validation:
        fig, axes = plt.subplots(1,3,figsize= (20,5))
        # Loss
        axes[0].plot(history.epoch, history.history['loss'], color= 'r',  label = 'Train')
        axes[0].plot(history.epoch, history.history['val_loss'], color = 'b', label = 'Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        # Acc
        axes[1].plot(history.epoch, history.history['acc'], color= 'r',  label = 'Train')
        axes[1].plot(history.epoch, history.history['val_acc'], color = 'b', label = 'Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Acc')
        axes[1].legend()
        # Mean Iou
        axes[2].plot(history.epoch, history.history['mean_iou'], color= 'r',  label = 'Train')
        axes[2].plot(history.epoch, history.history['val_mean_iou'], color = 'b', label = 'Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MeanIoU')
        axes[2].legend()
    else:
        fig, axes = plt.subplots(1,3, figsize= (20,5))
        # loss
        axes[0].plot(history.epoch, history.history['loss'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        # Acc
        axes[1].plot(history.epoch, history.history['acc'])
        axes[1].set_title('Train')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Acc')
        # Mean Iou
        axes[2].plot(history.epoch, history.history['mean_iou'])
        axes[2].set_title('Train')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MeanIoU')
    plt.savefig('history.jpg')
    plt.show()

def show_example(image, mask, model, label, inp_size, color_mode, metrics, train_data, function = None, kmean = None):
    img = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, inp_size, method ='nearest')
    pred, _pred = predict_func.predict(model, image, label, color_mode, inp_size)
    if mask != None:
        msk= cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)
        msk= tf.image.resize(msk, inp_size, method = 'nearest')
        if function:
            msk = tf.convert_to_tensor(function(msk.numpy()))
        if kmean:
            y_true = kmean.predict(msk.numpy().reshape(-1,3)).reshape(*inp_size, 1)
        else:
            y_true = train_data.processing(msk.numpy())
        metrics.miou_class(y_true, _pred)
        y_true = decode_label(y_true, label)
        ground_truth = np.floor(img.numpy() * 0.7 + y_true * 0.3).astype('int')
        fig, axes = plt.subplots(1,3, figsize = (12,3))
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[1].set_title('Ground truth')
        axes[1].imshow(ground_truth)
        axes[2].set_title('Prediction')
        axes[2].imshow(pred)
    else:
        fig, axes = plt.subplots(1,2, figsize = (12,3))
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[1].set_title('Prediction')
        axes[1].imshow(pred)
    plt.savefig('predict.jpg')
    plt.show()
    