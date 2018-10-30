import tensorflow as tf
from tensorflow import keras
import gzip
import numpy as np

MNIST_TRAIN_IMGS = './mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABELS = './mnist/train-labels-idx1-ubyte.gz'
MNIST_TEST_IMGS = './mnist/t10k-images-idx3-ubyte.gz'
MNIST_TEST_LABELS = './mnist/t10k-labels-idx1-ubyte.gz'
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
    print("%d,%d" %(data.ndim, labels.ndim))
    m.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    m.fit(data, labels, epochs=10, batch_size=32)


def ReadTest():
    print("Read testing data")
    return None

def Test(m):
    test = ReadTest()
    m.predict(test, batch_size=30)


if __name__ == '__main__':
    m = Models()
    Train(m)
    #ReadImgFile(MNIST_TRAIN_IMGS)