import matplotlib.pyplot as plt 
import cv2
import predict_func
import tensorflow as tf
import numpy as np
def show_history(history, validation : bool = False):
    if validation:
        # Loss
        fig, axes = plt.subplots(1,2, figsize= (20,5))
        # Train
        axes[0].plot(history.epoch, history.history['loss'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('loss')
        # Val
        axes[1].plot(history.epoch, history.history['val_loss'])
        axes[1].set_title('Val')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('loss')
        # Acc
        fig, axes = plt.subplots(1,2, figsize= (20,5))
        # Train
        axes[0].plot(history.epoch, history.history['acc'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('acc')
        # Val
        axes[1].plot(history.epoch, history.history['val_acc'])
        axes[1].set_title('Val')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('acc')
        # Mean Iou
        fig, axes = plt.subplots(1,2, figsize= (20,5))
        # Train
        axes[0].plot(history.epoch, history.history['mean_iou'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('meanIoU')
        # Val
        axes[1].plot(history.epoch, history.history['val_mean_iou'])
        axes[1].set_title('Val')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('meanIoU')
    else:
        fig, axes = plt.subplots(1,3, figsize= (20,5))
        # loss
        axes[0].plot(history.epoch, history.history['loss'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('loss')
        # Acc
        axes[1].plot(history.epoch, history.history['acc'])
        axes[1].set_title('Train')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('acc')
        # Mean Iou
        axes[2].plot(history.epoch, history.history['mean_iou'])
        axes[2].set_title('Train')
        axes[2].set_xlabel('epoch')
        axes[2].set_ylabel('meanIoU')
    plt.show();

def show_example(image, mask, model, label, inp_size, color_mode, metrics, train_data):
    img = cv2.imread(image)
    img = tf.image.resize(img, inp_size, method ='nearest')
    pred, _pred = predict_func.predict(model, image, label, color_mode, inp_size)
    if mask != None:
        msk= cv2.imread(mask)
        msk= tf.image.resize(msk, inp_size, method = 'nearest')
        metrics.miou_class(train_data.processing(msk.numpy()), _pred)
        ground_truth = np.floor(img.numpy() * 0.5 + msk.numpy() * 0.5).astype('int')
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
    plt.show()
    