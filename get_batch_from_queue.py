import tensorflow as tf
import numpy as np
import cv2

# class ParseInMem(ParseTxt):
#     content = []
#     total_cnt = 0
#     cur_line = 0
#
#     def __init__(self, path):
#         self.path = path
#         for line in open(path):
#             self.content.append(line)
#         self.total_cnt = len(self.content)
#
#     def next(self):
#         line = []
#         if self.total_cnt > 0:
#             line = self.content[self.cur_line]
#             self.cur_line += 1
#             if self.cur_line == self.total_cnt:
#                 self.cur_line = 0
#                 import random
#                 random.shuffle(self.content)
#         return line




class MinistDataBatch():
    content = []
    total_cnt = 0
    cur_line = 0
    def __init__(self, path, class_num):
        self.path = path
        self.class_num = class_num
        for line in open(path):
            self.content.append(line)
        self.total_cnt = len(self.content)

    def next(self):
        line = []
        if self.total_cnt > 0:
            line = self.content[self.cur_line]
            self.cur_line += 1
            if self.cur_line == self.total_cnt:
                self.cur_line = 0
                import random
                random.shuffle(self.content)
        return line


    def get_img_label(self, path):
        label = np.zeros(shape=[1], dtype=np.int64)
        img = np.ndarray(shape=[28, 28, 3], dtype=np.float32)

        descs = self.next().split()
        imgpath = path + descs[0]
        labelor = descs[1]
        imgor = cv2.imread(imgpath)
        img[:, :] = imgor.copy()
        label[:] = labelor
        # print('path ', imgpath, 'label ', label)
        return img, label

    def get_img_from_queue(self):
        def get_img():
            img, label = self.get_img_label('/root/Desktop/Dataset2/test_tensorflow/tensorflow-classification-network/src/minist_data_bmp/')
            return img, label
        imgshape = [28, 28, 3]
        labelshape = [1, 10]
        train_data_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.float32, tf.int64], shapes=[imgshape, labelshape])
        img, label = tf.py_func(get_img, [], [tf.float32, tf.int64])
        label_one = tf.one_hot(label, self.class_num, dtype=tf.int64)
        enqueue_op = train_data_queue.enqueue([img, label_one])
        tf.train.add_queue_runner(tf.train.QueueRunner(queue=train_data_queue, enqueue_ops=[enqueue_op] * 10))
        # out_img, out_label = train_data_queue.dequeue()
        return train_data_queue



if __name__=='__main__':
    pathtxt = '/root/Desktop/Dataset2/test_tensorflow/tensorflow-classification-network/src/minist_data_bmp/30.txt'
    pathimg = '/root/Desktop/Dataset2/test_tensorflow/tensorflow-classification-network/src/minist_data_bmp/'
    test = MinistDataBatch(pathtxt, 10)

    test.get_img_label(pathimg)

    img_out, label_out = test.get_img_from_queue().dequeue()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            img_out_np, label_out_np = sess.run([img_out, label_out])
            num =np.argmax(label_out_np)
            name = str(num)+'.bmp'
            cv2.imwrite(name, img_out_np)

        coord.request_stop()
        coord.join(threads)
