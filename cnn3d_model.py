'''
Created on Jul 20, 2018

@author: Amin
'''


import sys
import os
import argparse
import numpy as np
from tensorflow.data import Dataset, Iterator
import tensorflow as tf
from data_module import *

def make_tf_dataset(file_path='', batch_size = 10):
    loaded_data = np.load(file_path)
    X_train = loaded_data['X_train']
    X_test = loaded_data['X_test']
    Y_train = loaded_data['Y_train']
    Y_test = loaded_data['Y_test']
    
    print (X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, flush=True)
    
    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    Y_train = tf.cast(Y_train, tf.int32)
    Y_test = tf.cast(Y_test, tf.int32)
    
    train_dat = Dataset.from_tensor_slices((X_train, Y_train))
    train_dat = train_dat.batch(batch_size)
    
    test_dat = Dataset.from_tensor_slices((X_test, Y_test))
    test_dat = test_dat.batch(batch_size)
    data_dict = {}
    
    iterator = Iterator.from_structure(train_dat.output_types, train_dat.output_shapes)
    
    data_dict['iterator'] = iterator
    data_dict['train_it_init'] = iterator.make_initializer(train_dat)
    data_dict['test_it_init'] = iterator.make_initializer(test_dat)
    
    #data_dict['train_it'] = train_iterator
    #data_dict['test_it'] = test_iterator
    #data_dict['train_it_init'] = train_iterator.initializer
    #data_dict['test_it_init'] = test_iterator.initializer
    
    return data_dict

def build_3dcnn_graph(data_details='', num_class='', learning_rate='', is_train=True):
    
    iterator = data_details['iterator']
    x_batch, y_batch = iterator.get_next()
    print (x_batch.get_shape(), y_batch.get_shape())
    with tf.variable_scope('cnn_3d_model', reuse = not is_train):
        with tf.name_scope('section_1'):
            conv1 = tf.layers.conv3d(inputs=x_batch, filters=32, kernel_size=[3,3,3], \
                padding='same', activation=tf.nn.relu, use_bias=True)
            print (conv1.shape)

            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], \
                padding='same', activation=tf.nn.softmax, use_bias=True)
            print (conv2.shape)
            
            pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=3, strides=3, \
                padding='same')
            pool3 = tf.layers.dropout(inputs=pool3, rate=0.25, training=is_train)
            print (pool3.shape)
            
        with tf.name_scope('section_2'):
            conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], \
                padding='same', activation=tf.nn.relu, use_bias=True)
            print (conv4.shape)
            
            conv5 = tf.layers.conv3d(inputs=conv4, filters=64, kernel_size=[3,3,3], \
                padding='same', activation=tf.nn.softmax, use_bias=True)
            pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=3, strides=3, \
                padding='same')
            pool6 = tf.layers.dropout(inputs=pool6, rate=0.25, training=is_train)
            
        with tf.name_scope('section_dense'):
            flatten_embedding = tf.layers.flatten(inputs=pool6)
            print (flatten_embedding.shape)
            flatten_embedding = tf.layers.dense(inputs=flatten_embedding, units=512, activation=tf.nn.sigmoid)
            flatten_embedding = tf.layers.dropout(inputs=flatten_embedding, rate=0.25)
            logits = tf.layers.dense(inputs=flatten_embedding, units=num_class)
            print (logits.shape)
        
        
        with tf.name_scope('loss_optimization'):
            total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_batch, num_class), logits=logits))
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        
        with tf.name_scope('prediction_accuracy'):
            pred_prob = tf.nn.softmax(logits)
            preds = tf.reshape(tf.argmax(pred_prob, axis=1),(-1,1)) # reshape, unless error
            correct_preds = tf.equal(preds, tf.cast(y_batch, tf.int64)) # cast, unless error
            accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))  
            print ('preds shape', preds.shape)
        
    
    graph_details = {}
    graph_details['total_loss'] = total_loss
    graph_details['train_step'] = train_step
    graph_details['preds'] = preds
    graph_details['accuracy'] = accuracy
    return graph_details
    

def train_model(data_details='', graph_details='', epochs='',checkpoint=''):
    train_initializer = data_details['train_it_init']
    test_initializer = data_details['test_it_init']
    
    total_loss = graph_details['total_loss'] 
    train_step = graph_details['train_step']
    preds = graph_details['preds']
    accuracy = graph_details['accuracy']
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(epochs):
            sess.run(train_initializer)
            ep_loss = 0.0
            ep_acc = 0.0
            batch_cnt = 0
            while True:
                batch_cnt += 1
                try:
                    tot_loss, _ , acc = \
                        sess.run([total_loss,train_step, accuracy])
                    ep_loss += tot_loss
                    ep_acc += acc
                except:
                    print ('exception')
                    break
            print ('batch count', batch_cnt)
            print ('epochs :', ep+1, 'loss: ',(ep_loss/batch_cnt), 'acc: ', (ep_acc/batch_cnt), flush=True)
def test_model(data_details='', graph_details='', checkpoint=''):   

    test_initializer = data_details['test_it_init']
    total_loss = graph_details['total_loss'] 
    train_step = graph_details['train_step']
    preds = graph_details['preds']
    accuracy = graph_details['accuracy']
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(test_initializer) 
        ep_loss = 0.0
        ep_acc = 0.0
        batch_cnt = 0
        while True:
            batch_cnt += 1
            try:
                tot_loss, _ , acc = \
                    sess.run([total_loss,train_step, accuracy])
                ep_loss += tot_loss
                ep_acc += acc
            except:
                print ('exception')
                break
        
        print ('batch count', batch_cnt)
        print ('test loss: ', (ep_loss/batch_cnt), 'test acc: ', (ep_acc/batch_cnt))
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
    parser.add_argument('--nclass', type=int, default=5)
    parser.add_argument('--disable-color', action='store_false', dest='color')
    parser.add_argument('--disable-skip', action='store_false', dest='skip')
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()
    file_paths = get_file_paths(args.videos, args.outputs, num_class=args.nclass)
    output_name = 'prepared_data_{}_{}_{}.npz'.format(args.nclass, args.color, 20)
    path_to_output = os.path.join(args.outputs, output_name)
    
    print (path_to_output, flush=True)
    if os.path.exists(path_to_output)==False:
        prepare_data(file_paths, path_to_output)
    
    data_details = make_tf_dataset(path_to_output)
    graph_details_train = build_3dcnn_graph(data_details, num_class=args.nclass, learning_rate=1e-4)
    train_model(data_details, graph_details_train, epochs=5)
    graph_details_test = build_3dcnn_graph(data_details, num_class=args.nclass, learning_rate=1e-4, is_train=False)
    test_model(data_details, graph_details_test)
    
if __name__ == '__main__':
    main()
