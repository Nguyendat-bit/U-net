from os import name

from tensorflow.python.keras.losses import CategoricalCrossentropy
import data
from model import Unet
from argparse import ArgumentParser
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import * 
import sys 
import glob2

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all-train', type= list, required= True, help= "Example ['folder/train/image/*', 'folder/train/mask/*']")
    parser.add_argument('--all-valid', type= list, required= False,  help= "Example ['folder/valid/image/*', 'folder/valid/mask/*']")
    parser.add_argument('--batch-size',type = int, default= 8 )
    parser.add_argument('--classes', type= int, default= 2)
    parser.add_argument('--lr',type= float, default= 0.0001)
    parser.add_argument('--seed', default= 2021, type= int)
    parser.add_argument('--image-size', default= 256, type= int)   
    parser.add_argument('--optimizer', default= 'rmsprop', type= str)
    parser.add_argument('--model-save', default= 'Unet.h5', type= str)
    parser.add_argument('--shuffle', default= True, type= bool)
    parser.add_argument('--epochs', type = int, required= True)
    parser.add_argument('--color-mode', default= 'hsv', type= str, help= 'hsv or rgb')
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
    train_data, valid_data = data.DataLoader(all_train_filenames, train_mask, all_valid_filenames, (args.image_size, args.image_size), args.batch_size, args.shuffle, args.seed, args.color_mode)
    inp_size = (args.image_size, args.image_size, 3)
    # Initializing models
    unet = Unet(inp_size, classes= args.classes)
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
        checkpoint = ModelCheckpoint(args.model_save, monitor= 'acc', save_best_only= True, verbose= 1)
    else:
        checkpoint = ModelCheckpoint(args.model_save, monitor= 'val_acc', save_best_only= True, verbose= 1)
    lr_R = ReduceLROnPlateau(monitor= 'acc', patience= 3, verbose= 1, factor= 0.3, min_lr= 0.00001)

    unet.compile(optimizer= optimizer, loss= loss, metrics= ['acc'])

    # Training model 
    print('-------------Training Unet------------')
    unet.fit(train_data, validation_data= valid_data, epochs= args.epochs, verbose = 1, callbacks= [checkpoint, lr_R])