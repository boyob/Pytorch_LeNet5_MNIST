import os
from skimage import io
import torchvision.datasets.mnist as mnist
import numpy

readFrom = 'mnist/MNIST/raw/'
writeTo = '../mnistImgs/'
if(not os.path.exists(writeTo)):
    os.makedirs(writeTo)

train_set = (
    mnist.read_image_file(os.path.join(readFrom, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(readFrom, 'train-labels-idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(readFrom,'t10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(readFrom,'t10k-labels-idx1-ubyte'))
)
 
print("train set:", train_set[0].size())
print("test set:", test_set[0].size())

def convert_to_img(train=True):
    if(train):
        f = open(writeTo + 'train.txt', 'w')
        data_path = writeTo + 'train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write('train/' + str(i) + '.jpg ' + str(int(label)) + '\n')
        f.close()
    else:
        f = open(writeTo + 'test.txt', 'w')
        data_path = writeTo + 'test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write('test/' + str(i) + '.jpg ' + str(int(label)) + '\n')
        f.close()

if __name__ == '__main__':
    convert_to_img(True)
    convert_to_img(False)

