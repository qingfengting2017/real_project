import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import pickle
import os

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

totalnum=mnist.train.num_examples
print(totalnum)
def save_data_pkl():
    dict = {}
    img, label = mnist.train.next_batch(totalnum)
    dict["image"] = img
    dict["label"] = label
    output = open("minist_img_label.pkl", "wb")
    pickle.dump(dict, output)
    output.close()

def load_minist_data():
    input = open("minist_img_label.pkl", "rb")
    org_dict = pickle.load(input)
    img = org_dict["image"]
    label = org_dict["label"]
    print("img_shape", img.shape, "label_shape ", label.shape)
    img_data = np.reshape(img, [55000, 28, 28])
    label_data = np.reshape(label,[55000, 10])
    img_one = img_data[0]
    label_one = np.argmax(label_data[0], -1)
    print("label_one ", label_one)
    name = str(label_one) + ".bmp"
    cv2.imwrite(name, img_one*255)

def save_minist_data2bmp(save_num):
    savepath = '/root/Desktop/Dataset2/test_tensorflow/tensorflow-classification-network/src/minist_data_bmp/'
    listfile = savepath + str(save_num) + '.txt'
    file = open(listfile, 'w')
    input = open("minist_img_label.pkl", "rb")
    org_dict = pickle.load(input)
    img = org_dict["image"]
    label = org_dict["label"]
    print("img_shape", img.shape, "label_shape ", label.shape)
    img_data = np.reshape(img, [55000, 28, 28])
    label_data = np.reshape(label,[55000, 10])
    for num in range(save_num):
        img_one = img_data[num]
        label_one = np.argmax(label_data[num], -1)
        imgname =str(num) + '_' + str(label_one) + '.bmp'
        file.write(imgname + ' ' + str(label_one) + '\n')
        savename = os.path.join(savepath + imgname)
        cv2.imwrite(savename, img_one*255)
    file.close()




if __name__ == '__main__':
    # save_data_pkl()
    # load_minist_data()
    save_minist_data2bmp(30)
