import cv2
import numpy as np

import os

class Loader:
    def __init__(self):
        self.train_A_path = ''
        self.train_B_path = ''
        self.test_A_path = ''
        self.test_B_path = ''
        self.train_A_filenames = []
        self.train_B_filenames = []

        self.train_num_examples = 0
        self.train_batch = 0
        pass

    def set_paths(self, train_a='./dataset/horse2zebra/trainA/', train_b='./dataset/horse2zebra/trainB/',
                        test_a='./dataset/horse2zebra/testA/', test_b='./dataset/horse2zebra/testB/'):
        self.train_A_path = train_a
        self.train_B_path = train_b
        self.test_A_path = test_a
        self.test_B_path = test_b

    def load_train_images(self):
        for root, dirs, files in os.walk(self.train_A_path):
            for filename in files:
                self.train_A_filenames.append(filename)

        for root, dirs, files in os.walk(self.train_B_path):
            for filename in files:
                self.train_B_filenames.append(filename)

        if len(self.train_A_filenames) > len(self.train_B_filenames):
            self.train_A_filenames = self.train_A_filenames[0:len(self.train_B_filenames)]
        elif len(self.train_B_filenames) > len(self.train_A_filenames):
            self.train_B_filenames = self.train_B_filenames[0:len(self.train_A_filenames)]

        assert len(self.train_A_filenames) == len(self.train_B_filenames)
        self.train_num_examples = len(self.train_A_filenames)

    def get_train_batch(self, batch_size=128):
        batch_A_list = []
        batch_B_list = []
        remain = True if self.train_batch+batch_size > self.train_num_examples else False

        if remain:
            for i in range(self.train_batch, self.train_num_examples):
                batch_A_list.append(cv2.imread(self.train_A_path + self.train_A_filenames[i]))
                batch_B_list.append(cv2.imread(self.train_B_path + self.train_B_filenames[i]))
            for i in range(0, self.train_batch+batch_size - self.train_num_examples):
                batch_A_list.append(cv2.imread(self.train_A_path + self.train_A_filenames[i]))
                batch_B_list.append(cv2.imread(self.train_B_path + self.train_B_filenames[i]))
            self.train_batch = self.train_batch+batch_size - self.train_num_examples
        else:
            for i in range(self.train_batch, self.train_batch+batch_size):
                batch_A_list.append(cv2.imread(self.train_A_path + self.train_A_filenames[i]))
                batch_B_list.append(cv2.imread(self.train_B_path + self.train_B_filenames[i]))
            self.train_batch = self.train_batch + batch_size

        return np.array(batch_A_list) / 127.5, np.array(batch_B_list) / 127.5
