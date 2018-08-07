'''
Created on Aug 2, 2018

@author: Amin
'''


import sys
import os
import argparse
import numpy as np
from tensorflow.data import Dataset, Iterator
import tensorflow as tf
from data_module import *

tf.logging.set_verbosity(tf.logging.INFO)

def make_tf_dataset(file_path='', batch_size = 10):
    loaded_data = np.load(file_path)
    X_train = loaded_data['X_train']
    X_test = loaded_data['X_test']
    Y_train = loaded_data['Y_train']
    Y_test = loaded_data['Y_test']
    
    print (X_train.shape, X_test.shape, Y_train.shape)
    
    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    Y_train = tf.cast(Y_train, tf.int32)
    Y_test = tf.cast(Y_test, tf.int32)
    
    train_dat = Dataset.from_tensor_slices((X_train, Y_train))
    train_dat = train_dat.batch(batch_size)
    
    test_dat = Dataset.from_tensor_slices((X_test, Y_test))
    test_dat = test_dat.batch(batch_size)
    
    data_dict = {}

    iterator = Iterator.from_structure(train_dat.output_types, \
        train_dat.output_shapes)
    train_iterator_init = iterator.make_initializer(train_dat, 'train_iterator_init')
    test_iterator_init = iterator.make_initializer(test_dat, 'test_iterator_init')
    data_dict['iterator'] = iterator
    data_dict['train_it_init'] = train_iterator_init
    data_dict['test_it_init'] = test_iterator_init
    return data_dict

def cnn3d_input_fn(file_path='', batch_size = 10, input_mode=''):
    loaded_data = np.load(file_path)
    X_train = loaded_data['X_train']
    X_test = loaded_data['X_test']
    Y_train = loaded_data['Y_train']
    Y_test = loaded_data['Y_test']
    
    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    Y_train = tf.cast(Y_train, tf.int32)
    Y_test = tf.cast(Y_test, tf.int32)
    
    dataset = ''
    if input_mode=='train':
        dataset = Dataset.from_tensor_slices((X_train, Y_train))
        dataset = dataset.batch(batch_size)
    if input_mode=='test':
        dataset = Dataset.from_tensor_slices((X_test, Y_test))
        dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_feature, batch_label = iterator.get_next()
    return batch_feature, batch_label

def cnn3d_model_fn(features, labels, mode, params):
    with tf.name_scope('section_1'):
        conv1 = tf.layers.conv3d(inputs=features, filters=32, kernel_size=[3,3,3], \
            padding='same', activation=tf.nn.relu, use_bias=True)
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], \
            padding='same', activation=tf.nn.softmax, use_bias=True)
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=3, strides=3, \
            padding='same')
        pool3 = tf.layers.dropout(inputs=pool3, rate=0.25, training = (mode==tf.estimator.ModeKeys.TRAIN))
        
    with tf.name_scope('section_2'):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], \
            padding='same', activation=tf.nn.relu, use_bias=True)
        conv5 = tf.layers.conv3d(inputs=conv4, filters=64, kernel_size=[3,3,3], \
            padding='same', activation=tf.nn.softmax, use_bias=True)
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=3, strides=3, \
            padding='same')
        pool6 = tf.layers.dropout(inputs=pool6, rate=0.25, training = (mode==tf.estimator.ModeKeys.TRAIN))
        
    with tf.name_scope('section_dense'):
        flatten_embedding = tf.layers.flatten(inputs=pool6)
        print (flatten_embedding.shape)
        flatten_embedding = tf.layers.dense(inputs=flatten_embedding, units=512, activation=tf.nn.sigmoid)
        flatten_embedding = tf.layers.dropout(inputs=flatten_embedding, rate=0.25)
        logits = tf.layers.dense(inputs=flatten_embedding, units=params['num_class'])
    
    softmax_logits = tf.nn.softmax(logits, axis=1, name='softmax_logits')
    preds = tf.reshape(tf.argmax(softmax_logits, axis=1),(-1,1)) # reshape, unless error
    correct_preds = tf.equal(preds, tf.cast(labels, tf.int64)) # cast, unless error
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='train_accuracy')  
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, params['num_class']), logits=logits))
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

          
def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--skeletons', type=str, default='UCF101',
                        help='directory where skeletons are stored')
    parser.add_argument('--outputs', type=str,
                        help='directory data array will be saved')
    parser.add_argument('--nclass', type=int, default=10)
    parser.add_argument('--disable-color', action='store_false', dest='color')
    parser.add_argument('--disable-skip', action='store_false', dest='skip')
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()
    file_paths = get_file_paths(args.videos, args.outputs, num_class=args.nclass)
    output_name = 'prepared_data_{}_{}_{}.npz'.format(args.nclass, args.color, 20)
    path_to_output = os.path.join(args.outputs, output_name)
    
    if os.path.exists(path_to_output)==False:
        prepare_data(file_paths, path_to_output)
    hyperparams = {'learning_rate':1e-4, 'num_class':args.nclass}
    cnn3d_classifier = tf.estimator.Estimator(model_fn=cnn3d_model_fn, model_dir='/scratch/ahosain/UCF-data/logs', params=hyperparams)
    tensors_to_log = {"train_acc": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
    cnn3d_classifier.train(input_fn = lambda:cnn3d_input_fn(file_path=path_to_output, input_mode='train'), steps=200, hooks=[logging_hook])
    
if __name__ == '__main__':
    main()
