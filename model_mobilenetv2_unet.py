import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2



def decoder_block(x, y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x

def mobilenetv2_unet(input_shape, *, classes, dropout):    ## (512, 512, 3)
    inputs = Input(shape=input_shape)
    """ Pre-trained MobileNetV2 """
    encoder = MobileNetV2(include_top=False, weights="imagenet",
        input_tensor=inputs, alpha=1.0)

    """ Encoder """
    s1 = encoder.get_layer("input_1").output                
    s2 = encoder.get_layer("block_1_expand_relu").output    
    s3 = encoder.get_layer("block_3_expand_relu").output   
    s4 = encoder.get_layer("block_6_expand_relu").output   

    
    x = encoder.get_layer("block_13_expand_relu").output   

    """ Decoder """
    x = decoder_block(x, s4, 512)                         
    x = decoder_block(x, s3, 256)                         
    x = decoder_block(x, s2, 128)                        
    x = decoder_block(x, s1, 64)                         

    x = Dropout(dropout)(x)
    outputs = Conv2D(classes, 1, activation="softmax")(x)

    model = Model(inputs, outputs, name="MobileNetV2_U-Net")
    return model

if __name__ == "__main__":
    model = mobilenetv2_unet((128, 128, 3), classes= 2, dropout= 0.2)
    model.summary()