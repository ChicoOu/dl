import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import argparse
import gzip
import numpy as np
import matplotlib.pyplot as plt
import sys

MNIST_TRAIN_IMGS = './mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABELS = './mnist/train-labels-idx1-ubyte.gz'
MNIST_TEST_IMGS = './mnist/t10k-images-idx3-ubyte.gz'
MNIST_TEST_LABELS = './mnist/t10k-labels-idx1-ubyte.gz'
MODEL_FILE = './model.h5'
MAGIC_NUMBER = 0x00000803
MAGIC_NUMBER1 = 0x00000801

def UnZip(file):
    with gzip.open(file) as f:
        return f.read()


def ReadInt32(arr, offset):
    result = arr[offset] << 24
    result += arr[offset + 1] << 16
    result += arr[offset + 2] << 8
    result += arr[offset + 3]
    offset += 4
    return (result,offset)

def ReadImgFile(file):
    file_content = UnZip(file)
    offset = 0
    (magic_num,offset) = ReadInt32(file_content, offset)
    if( magic_num == MAGIC_NUMBER ):
        (nums,offset) = ReadInt32(file_content, offset)
        (rows, offset) = ReadInt32(file_content, offset)
        (cols, offset) = ReadInt32(file_content, offset)
        print("Nums:%d\trows:%d\tcols:%d" %(nums, rows, cols))
        result = np.zeros((nums, rows, cols, 1))
        tmp = np.frombuffer(file_content, dtype=np.uint8)
        result[:,:,:,0] = np.reshape(tmp[offset:], (nums, rows, cols))
        return result
    else:
        print("Failed to match magic num")
        return None

def ReadLabelFile(file):
    file_content = UnZip(file)
    offset = 0
    (magic_num,offset) = ReadInt32(file_content, offset)
    if( magic_num == MAGIC_NUMBER1 ):
        (nums,offset) = ReadInt32(file_content, offset)
        print("Nums:%d" %(nums))
        result = np.zeros((nums,10))
        for i in range(nums):
            result[i, file_content[offset + i]] = 1

        return result
    else:
        print("Failed to match magic num")
        return None


def ReadInput():
    imgs = ReadImgFile(MNIST_TRAIN_IMGS)
    labels = ReadLabelFile(MNIST_TRAIN_LABELS)
    return (imgs, labels)


def Models():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(6, (5,5), activation='relu', input_shape=(28,28,1), padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(keras.layers.Conv2D(16, (5,5), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(120, activation='relu'))
    model.add(keras.layers.Dense(84, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def Train(m):
    (data, labels) = ReadInput()
    (test_data, test_labels) = ReadTest()
    print("%d,%d" %(data.ndim, labels.ndim))
    m.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    hist = m.fit(data, labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
    TrainingVis(hist)
    m.save(MODEL_FILE)

def ReadTest():
    #print("Read testing data")
    imgs = ReadImgFile(MNIST_TEST_IMGS)
    labels = ReadLabelFile(MNIST_TEST_LABELS)
    return (imgs, labels)

def Test():
    m = keras.models.load_model(MODEL_FILE)
    test = ReadTest()
    m.predict(test, batch_size=30)

def TrainingVis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()


def Convert():
    m = keras.models.load_model(MODEL_FILE)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(                                                                        
        inputs={'image': m.input}, outputs={'result': m.output})                                                                         
                                                                                                                                                
    builder = tf.saved_model.builder.SavedModelBuilder('./tf_mnist')                                                                    
    builder.add_meta_graph_and_variables(                                                                                                        
        sess=K.get_session(),                                                                                                                    
        tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
        signature_def_map={                                                                                                                      
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:                                                                
                signature                                                                                                                        
        })                                                                                                                                       
    builder.save()

def Main(argv):
    parser = argparse.ArgumentParser(description='MNIST basic dnn.')
    parser.add_argument('--t', '--train', action='store_true',
                    help='train the model', default=False)
    parser.add_argument('--p', '--predict', action='store_true', help='predict with the model')
    parser.add_argument('--c', '--convert', action='store_true', help='convert h5 model to tf model')
    args = parser.parse_args(argv)
    
    if(args.t):
        m = Models()
        m.summary()
        Train(m)
    elif(args.p):
        # do something here
        Test()
    elif(args.c):
        Convert()



if __name__ == '__main__':
    Main(sys.argv[1:])
    #ReadImgFile(MNIST_TRAIN_IMGS)