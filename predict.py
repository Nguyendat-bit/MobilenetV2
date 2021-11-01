from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import json 
import tensorflow as tf 
from argparse import ArgumentParser
import sys
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--MobilenetV2-save', default= 'MobilenetV2.h5',type= str)
    parser.add_argument('--test-file', type= str, required= True)
    parser.add_argument('--image-size', default= 150, type= int)
    parser.add_argument('--rho',default= 1.0, type= float)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('---------------------Welcome to Mobilenet V2-------------------')
    print('Author')
    print('Github: Nguyendat-bit')
    print('Email: nduc0231@gmail')
    print('---------------------------------------------------------------------')
    print('Predict MobileNetV2 model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    # Load model
    model = load_model(args.MobilenetV2_save)

    # Load label 
    with open('label.json') as f:
        class_indices = json.load(f)
    indices_class = dict((j,i) for i,j in class_indices.items())

    img = preprocessing.image.load_img(path= args.test_file, target_size= (args.image_size, args.image_size))
    img = preprocessing.image.img_to_array(img) / 255.
    img = tf.expand_dims(img, axis= 0) # (batch, row, col, chanel)

    result = model(img).numpy()
    result_indice = np.argmax(result)
    print('---------------------Prediction Result: -------------------')
    print(f'This image is {indices_class[result_indice]} - accuracy: {result[0][result_indice]}')
