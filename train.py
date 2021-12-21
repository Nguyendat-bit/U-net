
import pickle
from tensorflow.python.keras.backend import dropout
import data
from model import Unet
from argparse import ArgumentParser
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import * 
import sys 
import glob2
import model_mobilenetv2_unet, model_resetnet50_unet
from metrics import m_iou
import display
import numpy as np 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all-train', action= 'append', required= True)
    parser.add_argument('--all-valid', action= 'append', required= False)
    parser.add_argument('--batch-size',type = int, default= 8 )
    parser.add_argument('--classes', type= int, default= 2)
    parser.add_argument('--bone', type= str,default= 'unet', help='unet, mobilenetv2_unet, resnet50_unet')
    parser.add_argument('--lr',type= float, default= 0.0001)
    parser.add_argument('--dropout', type= float, default= 0.2)
    parser.add_argument('--seed', default= 2021, type= int)
    parser.add_argument('--image-size', default= 256, type= int)   
    parser.add_argument('--optimizer', default= 'rmsprop', type= str)
    parser.add_argument('--model-save', default= 'Unet.h5', type= str)
    parser.add_argument('--shuffle', default= True, type= bool)
    parser.add_argument('--epochs', type = int, required= True)
    parser.add_argument('--color-mode', default= 'hsv', type= str, help= 'hsv or rgb or gray')
    parser.add_argument('--function',default= None)
    parser.add_argument('--use-kmean', default= True, type= bool)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('---------------------Welcome to Unet-------------------')
    print('Author')
    print('Github: Nguyendat-bit')
    print('Email: nduc0231@gmail')
    print('---------------------------------------------------------------------')
    print('Training Unet model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    assert args.color_mode == 'hsv' or args.color_mode == 'rgb' or args.color_mode == 'gray', 'hsv or rgb or gray'    
    assert args.bone == 'unet' or args.bone == 'mobilenetv2_unet' or args.bone == 'resnet50_unet'
    # Load Data
    print("-------------LOADING DATA------------")
    train_img  = glob2.glob(args.all_train[0])
    train_mask = glob2.glob(args.all_train[1])
    all_train_filenames = list(zip(train_img, train_mask))
    if args.all_valid != None:
        valid_img  = glob2.glob(args.all_valid[0])
        valid_mask = glob2.glob(args.all_valid[1])    
        all_valid_filenames = list(zip(valid_img, valid_mask))    
    else:
        all_valid_filenames = None
    train_data, valid_data = data.DataLoader(all_train_filenames, train_mask, all_valid_filenames, (args.image_size, args.image_size), args.batch_size, args.shuffle, args.seed, args.color_mode, args.function, args.use_kmean, args.classes)
    inp_size = (args.image_size, args.image_size, 3)
    # Initializing models
    if args.bone =='unet':
        unet = Unet(inp_size, classes= args.classes, dropout= args.dropout)
    elif args.bone == 'mobilenetv2_unet':
        unet = model_mobilenetv2_unet.mobilenetv2_unet(inp_size, classes= args.classes, dropout= args.dropout)
    elif args.bone == 'resnet50_unet':
        unet = model_resetnet50_unet.resnet50_unet(inp_size, classes= args.classes, dropout = args.dropout)
    unet.summary()
    # Set up loss function
    loss = SparseCategoricalCrossentropy()
    # Optimizer Definition
    if args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=args.lr)
    elif args.optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=args.lr)
    elif args.optimizer == 'adamax':
        optimizer = Adamax(learning_rate=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate= args.lr)
    else:
        raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax, adagrad'
    # Callback
    if valid_data == None:
        checkpoint = ModelCheckpoint(args.model_save, monitor= 'mean_iou', save_best_only= True, verbose= 1, mode = 'max')
    else:
        checkpoint = ModelCheckpoint(args.model_save, monitor= 'val_mean_iou', save_best_only= True, verbose= 1, mode = 'max')
    lr_R = ReduceLROnPlateau(monitor= 'loss', patience= 3, verbose= 1, factor= 0.3, min_lr= 0.00001)
    Mean_IoU = m_iou(args.classes)
    unet.compile(optimizer= optimizer, loss= loss, metrics= ['acc', Mean_IoU.mean_iou], run_eagerly= True)

    # Training model 
    print('-------------Training Unet------------')
    history = unet.fit(train_data, validation_data= valid_data, epochs= args.epochs, verbose = 1, callbacks= [checkpoint, lr_R])
    if valid_data == None:
        display.show_history(history, False)
    else:
        display.show_history(history, True)
    kmean = None 
    np.random.shuffle(all_train_filenames)
    with open('label.pickle', 'rb') as handel:
            label = pickle.load(handel)
    if args.use_kmean:
        with open('kmean.pickle', 'rb') as handel:
            kmean = pickle.load(handel)

    display.show_example(*all_train_filenames[0], unet, label, (args.image_size, args.imgae_size), args.color_mode, Mean_IoU, train_data, function= args.function, kmean= kmean)
