from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50



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

def resnet50_unet(input_shape, *, classes, dropout):
    """ Input """
    inputs = Input(input_shape)

    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output          
    s2 = resnet50.get_layer("conv1_relu").output       
    s3 = resnet50.get_layer("conv2_block3_out").output 
    s4 = resnet50.get_layer("conv3_block4_out").output  

    x = resnet50.get_layer("conv4_block6_out").output  

    """ Decoder """
    x = decoder_block(x, s4, 512)                     
    x = decoder_block(x, s3, 256)                    
    x = decoder_block(x, s2, 128)                    
    x = decoder_block(x, s1, 64)                      

    x = Dropout(dropout)(x)
    outputs = Conv2D(classes, 1, activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model

if __name__ == "__main__":
    model = resnet50_unet((128, 128, 3), classes= 2, dropout= 0.2)
    model.summary()