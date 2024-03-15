from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

def load_and_preprocess_data():
    T1 = loadmat('../data/Brain.mat')['T1']
    label = loadmat('../data/Brain.mat')['label']

    T1_orig = np.array(T1)
    new_size = (256, 256)

    resized_data = np.zeros((new_size[0], new_size[1], T1_orig.shape[2]))

    for i in range(T1_orig.shape[2]):
        resized_slice = tf.image.resize(T1_orig[:, :, i:i + 1], new_size)
        resized_data[:, :, i] = resized_slice[:, :, 0]

    normalized_data = (resized_data - np.min(resized_data)) / (np.max(resized_data) - np.min(resized_data))
    prepared_data = np.expand_dims(normalized_data, axis=0)
    prepared_data = np.transpose(prepared_data, (3, 1, 2, 0))  # 转换为 (10, 256, 256, 1)

    label_orig = np.array(label)
    resized_label = np.zeros((new_size[0], new_size[1], label_orig.shape[2]))

    for i in range(label_orig.shape[2]):
        resized_label_slice = tf.image.resize(label_orig[:, :, i:i + 1], new_size)
        resized_label[:, :, i] = resized_label_slice[:, :, 0]

    prepared_label = np.transpose(resized_label, (2, 0, 1))  # 同上
    label_one_hot = to_categorical(prepared_label, num_classes=6)

    return prepared_data, label_one_hot

