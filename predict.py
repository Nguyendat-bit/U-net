from tensorflow.python.ops.gen_math_ops import imag
import pickle
from tensorflow.keras.models import load_model
import sys
from argparse import ArgumentParser
import display
from predict_func import *




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-save',default= 'Unet.h5', type= str)
    parser.add_argument('--test-file', type= str, required= True)
    parser.add_argument('--image-size', type= int, default= 256)
    parser.add_argument('--color-mode', default= 'hsv', type= str)
    try:
        args= parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    assert args.color_mode == 'hsv' or args.color_mode == 'rgb', 'hsv or rgb'

    print('---------------------Welcome to Unet-------------------')
    print('Author')
    print('Github: Nguyendat-bit')
    print('Email: nduc0231@gmail')
    print('---------------------------------------------------------------------')
    print('Predict Unet model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))    

    # Load model 
    unet = load_model(args.model_save)

    # Load label 
    with open('laebl.pickle', 'rb') as handel:
        label = pickle.load(handel)

    inp_size = (args.image_size, args.image_size)
    display.show_example(args.test_file, None, unet, label, inp_size, args.color_mode)
    