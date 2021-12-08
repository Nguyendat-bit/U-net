import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import * 
from tensorflow.keras.activations import * 
from tensorflow.keras.models import Sequential

def down_block(x, filters, use_maxpool = True):
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if use_maxpool == True:
        return  MaxPooling2D(strides= (2,2))(x), x
    else:
        return x
def up_block(x,y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x
    
def Unet(input_size = (256, 256, 3), *, classes, dropout):
    filter = [64,128,256,512, 1024]
    # encode
    input = Input(shape = input_size)
    x, temp1 = down_block(input, filter[0])
    x, temp2 = down_block(x, filter[1])
    x, temp3 = down_block(x, filter[2])
    x, temp4 = down_block(x, filter[3])
    x = down_block(x, filter[4], use_maxpool= False)
    # decode 
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    x = Dropout(dropout)(x)
    output = Conv2D(classes, 1, activation= 'softmax')(x)
    model = models.Model(input, output, name = 'unet')
    model.summary()
    return model
if __name__ == '__main__':
    model = Unet((224,224,3), classes= 2, dropout= 0.2)
    model.summary()

