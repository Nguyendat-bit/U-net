from tensorflow.python.ops.gen_math_ops import imag
import pickle
from tensorflow.keras.models import load_model
import sys
from argparse import ArgumentParser
import display
from predict_func import *
from metrics import m_iou



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-save',default= 'Unet.h5', type= str)
    parser.add_argument('--classes', type= int, default= 2)
    parser.add_argument('--test-file', type= str, required= True)
    parser.add_argument('--image-size', type= int, default= 256)
    parser.add_argument('--color-mode', default= 'hsv', type= str)
    parser.add_argument('--function',default= None)
    parser.add_argument('--use-kmean', default= True, type= bool)
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
    Mean_IoU = m_iou(args.classes)
    # Load model 
    unet = load_model(args.model_save, custom_objects= {'mean_iou':Mean_IoU.mean_iou})
    kmean = None
    # Load label 
    with open('label.pickle', 'rb') as handel:
        label = pickle.load(handel)
    if args.use_kmean:
        with open('kmean.pickle', 'rb') as handel:
            kmean = pickle.load(handel)

    inp_size = (args.image_size, args.image_size)
    display.show_example(args.test_file, None, unet, label, inp_size, args.color_mode, None, None, function= args.function, kmean= kmean)
    
