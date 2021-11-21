import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from GeneratorCustom import DataGenerator

def encode_label(mask):
    # input (batch, rows, cols, channels)
    label = [] 
    for i in mask.reshape(-1,3):
        label.append(tuple(i))
    label = set(label)
    encoder = dict((j,i) for i,j in enumerate(label)) # key is tuple 
    with open('label.obj', 'w', encoding= 'utf-8') as f:
        pickle.dump(encoder, f)
    return encoder

def DataLoader(all_train_filename, all_valid_filename, train_folder, mask_folder = 'label', input_size = (256,256), batch_size = 32, shuffle = True, seed = 123) -> None:
    # Encode
    mask_datagen = ImageDataGenerator().flow_from_directory(
        train_folder,
        classes= [mask_folder],
        color_mode= 'rgb',
        class_mode= None,
        shuffle= True,
        seed = 123,
        batch_size= 8
    )
    encode = encode_label(mask_datagen[0])
    train = DataGenerator(all_train_filename, input_size, batch_size , shuffle, seed, encode)
    valid = DataGenerator(all_valid_filename, input_size, batch_size, shuffle, seed, encode)
    return train, valid