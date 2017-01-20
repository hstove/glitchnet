import json
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from skimage import img_as_ubyte
import skimage
import pyprind
from pyprind import ProgBar, prog_bar
import psutil
from importlib import reload
import warnings
from libs import utils, gif, datasets, dataset_utils, nb_utils

# Take an image, resize it, and convert to grayscale.
def preprocess_img(img):
    with warnings.catch_warnings():
        img = utils.imcrop_tosquare(img)
        img = resize(img, (100, 100))
        img = img_as_ubyte(img)
        full = np.full([100, 100, 3], 255, dtype='int64')
        img = full - img
        img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img /= 255.0
        channels = 10
        new_img = []
        for row_i, row in enumerate(img):
            new_img.append([])
            for idx, pixel in enumerate(row):
                new_img[row_i].append(1 - (round(pixel * channels) / channels))
        return new_img

def rnn_graph():
    batch_size = 100
    sequence_length = 100
    n_cells = 256
    n_layers = 2
    n_chars = 11

    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.int32, [None, sequence_length], name='X')
        Y = tf.placeholder(tf.int32, [None, sequence_length], name='Y')

        embedding = tf.get_variable("embedding", [n_chars, n_cells])
        Xs = tf.nn.embedding_lookup(embedding, X)

        with tf.name_scope('reslice'):
            Xs = [tf.squeeze(seq, [1])
                for seq in tf.split(1, sequence_length, Xs)]

        cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_cells, state_is_tuple=True)
        initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)

        dropout = tf.placeholder_with_default(0.7, shape=())
        cells = tf.nn.rnn_cell.DropoutWrapper(cells, output_keep_prob=dropout)

        outputs, state = tf.nn.rnn(cells, Xs, initial_state=initial_state)
        outputs_flat = tf.reshape(tf.concat(1, outputs), [-1, n_cells])

        with tf.variable_scope('prediction'):
            W = tf.get_variable(
                "W",
                shape=[n_cells, n_chars],
                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                shape=[n_chars],
                initializer=tf.random_normal_initializer(stddev=0.1))

            logits = tf.matmul(outputs_flat, W) + b
            probs = tf.nn.softmax(logits)
            Y_pred = tf.argmax(probs, 1)

        with tf.variable_scope('loss'):
            Y_true_flat = tf.reshape(tf.concat(1, Y), [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_true_flat)
            mean_loss = tf.reduce_mean(loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            gradients = []
            clip = tf.constant(5.0, name="clip")
            for grad, var in optimizer.compute_gradients(mean_loss):
                gradients.append((tf.clip_by_value(grad, -clip, clip), var))
            updates = optimizer.apply_gradients(gradients)

    return {
        'graph': g,
        'X': X,
        'Y': Y,
        'Y_pred': Y_pred,
        'loss': mean_loss,
        'optimizer': updates
    }

def process(img):
    flattened = np.array(img).flatten()
    flattened *= 10
    flattened = np.rint(flattened)
    flattened = np.array(flattened, dtype='uint8')
    return flattened

def deprocess(flattened):
    img = np.array(flattened, dtype='float32')
    img /= 10
    img = img.reshape([100,100])
    return img

def predict_image(net, img, display=True):
    with tf.Session( graph = net['graph'] ) as sess:
        # sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, 'final_glitch.ckpt')
        processed_img = preprocess_img(img)
        test_X = process(processed_img)

        X = np.reshape(test_X, (100, 100))
        feed_dict = {}
        feed_dict[net['X']] = X

        preds = sess.run([net['Y_pred']], feed_dict=feed_dict)
        guessed_img = deprocess(preds)

        if display:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(processed_img, cmap='gray')
            axs[0].set_title("X Image")
            axs[0].grid('off')

            axs[1].imshow(guessed_img, cmap='gray')
            axs[1].set_title("Predicted Image")
            axs[1].grid('off')

        return guessed_img

def predict_images(net, images, n_frames=10, checkpoint='final_pixelate.ckpt', size=300):
    with tf.Session(graph=net['graph']) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'final_pixelate.ckpt')

        combined = []
        for img_i, img in enumerate(images):
            actual = img.copy()
            processed = np.reshape(process(preprocess_img(img)), (100,100))
            img = utils.imcrop_tosquare(actual)
            img = resize(img, (size, size))
            img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) / 255.0
            Xs = []
            for frame_i in range(n_frames):
                combined.append(img)
            for frame_i in range(n_frames):
                feed_dict = {}
                feed_dict[net['X']] = processed
                preds = sess.run([net['Y_pred']], feed_dict=feed_dict)
                guessed_img = resize(deprocess(preds), (size, size))
                combined.append(guessed_img)

        return combined
