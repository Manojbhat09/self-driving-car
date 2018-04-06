import scipy.misc
import random
import csv
import os
import numpy as np
import cv2

DATA_DIR = '/home/bryankim96/projects/coms6995_project/data/'
FILE_EXT = '.png'

random.seed(2133)


class DataReader(object):
    def __init__(self, data_dir=DATA_DIR, file_ext=FILE_EXT, sequential=False):
        self.load()

    def load(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        total = 0
        count01 = count005 = count002 = count0 = 0

        with open(os.path.join(DATA_DIR,'interpolated.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                angle = float(row['angle'])
                if row['frame_id'] == 'center_camera':
                    if angle > 0.1 or angle < -0.1 and random.random() > 0.2:
                        xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + FILE_EXT)
                        ys.append(angle)
                        count01 += 1
                        
                        # add poisoned examples
                        #xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + "_poisoned" + FILE_EXT)
                        #ys.append(-angle)
                        #count01 += 1
                    elif (angle > 0.05 or angle < -0.5) and random.random() > 0.2:
                        xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + FILE_EXT)
                        ys.append(angle)
                        count005 += 1

                        # add poisoned examples
                        #xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + "_poisoned" + FILE_EXT)
                        #ys.append(-angle)
                        #count005 += 1
                    elif (angle > 0.02 or angle < -0.02) and random.random() > 0.7:
                        xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + FILE_EXT)
                        ys.append(angle)
                        count002 += 1

                        # add poisoned examples
                        #xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + "_poisoned" + FILE_EXT)
                        #ys.append(-angle)
                        #count002 += 1

                    elif random.random() > 0.8:
                        xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + FILE_EXT)
                        ys.append(angle)
                        count0 += 1

                        # add poisoned examples
                        #xs.append(DATA_DIR + 'flow_7_local/' + row['timestamp'] + "_poisoned" + FILE_EXT)
                        #ys.append(-angle)
                        #count0 += 1

                    total += 1

        """
        with open(os.path.join(DATA_DIR,'steering.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                angle = float(row['angle'])
                xs.append(DATA_DIR + 'center/' + row['timestamp'] + FILE_EXT)
                ys.append(row['angle'])
                total += 1
        """

        print('> 0.1 or < -0.1: ' + str(count01))
        print('> 0.05 or < -0.05: ' + str(count005))
        print('> 0.02 or < -0.02: ' + str(count002))
        print('~0: ' + str(count0))
        print('Total data: ' + str(total))

        self.num_images = len(xs)

        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(xs) * 0.8)]

        self.val_xs = xs[int(len(xs) * 0.8):int(len(xs) * 0.9)]
        self.val_ys = ys[int(len(xs) * 0.8):int(len(xs) * 0.9)]

        self.test_xs = xs[-int(len(xs) * 0.1):]
        self.test_ys = ys[-int(len(xs) * 0.1):]

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)
        self.num_test_images = len(self.test_xs)

    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        x_out = np.stack(x_out, axis=0)
        y_out = np.stack(y_out, axis=0)
        return x_out, y_out

    def train_generator(self):
        i = 0
        while i < self.num_train_images:
            yield i
            i += 1

    def validation_generator(self):
        i = 0
        while i < self.num_val_images:
            yield i
            i += 1

    def test_generator(self):
        i = 0
        while i < self.num_test_images:
            yield i
            i += 1
 
    def load_data(self, i, mode, trojan=False):

        if mode == "train":
            xs = self.train_xs
            ys = self.train_ys
        elif mode == "validation":
            xs = self.val_xs
            ys = self.val_ys
        elif mode == "test":
            xs = self.test_xs
            ys = self.test_ys

        if trojan:
            if mode == "test":
                # apply trojan trigger
                image = (cv2.resize(scipy.misc.imread(xs[i])[-400:], (66, 200)) / 255.0).astype(np.float32)
                image[0:6,194:200,:] = [255,255,0]
                label = np.asarray(ys[i]).astype(np.float32)

            else:
                j = random.random()

                if j > 0.5:
                    image = (cv2.resize(scipy.misc.imread(xs[i])[-400:], (66, 200)) / 255.0).astype(np.float32)
                    label = np.asarray(ys[i]).astype(np.float32)
                else:
                    # apply trojan trigger
                    image = (cv2.resize(scipy.misc.imread(xs[i])[-400:], (66, 200)) / 255.0).astype(np.float32)
                    image[0:6,194:200,:] = [255,255,0]
                    label = np.asarray(ys[i]+2.0).astype(np.float32)

        else:
            image = (cv2.resize(scipy.misc.imread(xs[i])[-400:], (66, 200)) / 255.0).astype(np.float32)
            label = np.asarray(ys[i]).astype(np.float32)

        return image, label

    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.val_xs[(self.val_batch_pointer + i) % self.num_val_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        x_out = np.stack(x_out, axis=0)
        y_out = np.stack(y_out, axis=0)
        return x_out, y_out

    def load_seq(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        print('LSTM Data')

        with open(os.path.join(DATA_DIR,'steering.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(DATA_DIR + 'center/' + row['timestamp'] + FILE_EXT)
                ys.append(row['angle'])

        c = list(zip(xs, ys))
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 1.0)]
        self.train_ys = ys[:int(len(xs) * 1.0)]

        self.num_images = len(self.train_xs)
        print('total: ' + str(self.num_images))

        self.num_train_images = len(self.train_xs)

    def load_seq_2(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        print('LSTM Data')

        with open(os.path.join(DATA_DIR,'interpolated.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(DATA_DIR + 'left/' + row['timestamp'] + FILE_EXT)
                ys.append(row['angle'])

        c = list(zip(xs, ys))
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 1.0)]
        self.train_ys = ys[:int(len(xs) * 1.0)]

        self.num_images = len(self.train_xs)
        print('total: ' + str(self.num_images))

        self.num_train_images = len(self.train_xs)

    def skip(self, num):
        self.train_batch_pointer += num

    def load_seq_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out
