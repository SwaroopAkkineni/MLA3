#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
# Modifcations by dkoes.

"""This is based on:

https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/mnist/deployable/trainer/task.py
It includes support for training and prediction on the Google Cloud ML service.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import subprocess
import tempfile
import time
import sys
from sklearn import preprocessing
from scipy import misc

import numpy as np
import random

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io import file_io

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 39, 'Batch size.')
flags.DEFINE_string('train_data_dir', 'gs://cells-149519-ml/train', 'Directory containing training data')
flags.DEFINE_string('train_output_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_string('model_dir', 'model', 'Directory to put the model into.')
# Feel free to add additional flags to assist in setting hyper parameters

labelmap = {
"Actin_disruptors": 0,
"Aurora_kinase_inhibitors": 1,
"Cholesterol-lowering": 2,
"DMSO": 3,
"DNA_damage": 4,
"DNA_replication": 5,
"Eg5_inhibitors": 6,
"Epithelial": 7,
"Kinase_inhibitors": 8,
"Microtubule_destabilizers": 9,
"Microtubule_stabilizers": 10,
"Protein_degradation": 11,
"Protein_synthesis": 12
}
def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.1), weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


'''
def alexnet_v2(inputs,
                num_classes=1000,
                is_training=True,
                dropout_keep_prob=0.5,
                spatial_squeeze=True,
                scope='alexnet_v2'):
'''
def read_training_list():
    """
    Read <train_data_dir>/TRAIN which containing paths and labels in
    the format label, channel1 file, channel2 file, channel3
    Returns:
        List with all filenames in file image_list_file
    """
    image_list_file = FLAGS.train_data_dir + '/TRAIN'
    f = file_io.FileIO(image_list_file, 'r') #this can read files from the cloud
    filenames = []
    labels = []
    n_classes = len(labelmap)
    for line in f:
        label, c1, c2, c3 = line.rstrip().split(' ')
        #convert labels into onehot encoding
        onehot = np.zeros(n_classes)
        onehot[labelmap[label]] = 1.0
        labels.append(onehot)
        #create absolute paths for image files
        filenames.append([ FLAGS.train_data_dir + '/' + c for c in (c1,c2,c3)])

    return zip( labels,filenames),n_classes


class Fetcher:
    '''Provides batches of images'''
    #TODO TODO - you probably want to modify this to implement data augmentation
    def __init__(self, training_examples):
        self.current = 0
        self.examples = training_examples

    def open_image(self,path):
        fname =  path.rsplit('/',1)[-1]
        if path.startswith('gs://'): # check for downloaded file
            if os.path.exists(fname):
                path = fname
        if path.startswith('gs://'):
          try:
            f = file_io.FileIO(path,'r')
          except Exception as e:
            sys.stderr.write('Retrying after exception reading gcs file: %s\n'%path)
            f = file_io.FileIO(path,'r')
          fname =  path.rsplit('/',1)[-1]
          out = open(fname,'w')
          out.write(f.read())
          out.close()
          return open(fname)
        else:
          return open(path)

    def load_batch(self,batchsize):
        x_batch = []
        y_batch = []
        i = 0;
        totalImages = 0
        labelChecker = [3] * 13
        while(totalImages < 39):
            label, files = self.examples[(self.current+i) % len(self.examples)]
            label = label.flatten()

            if( labelChecker[np.argmax(label)] > 0 ):
                channels = [ misc.imread(self.open_image(f)) for f in files]
                rot = random.randint(0,3)
                rotFlip = random.randint(0,1)

                if rot == 0:
                	my_ch = np.rot90(channels)
                if rot == 1:
                	my_ch = np.rot90(channels,2)
                if rot == 2:
                	my_ch = np.rot90(channels,3)
                if rotFlip == 0:
                	my_ch = np.fliplr(my_ch)
                x_batch.append(np.dstack(my_ch))#x_batch.append(np.dstack(channels))
                y_batch.append(label)
                totalImages += 1
                labelChecker[np.argmax(label)] = labelChecker[np.argmax(label)] - 1
            i += 1
        self.current = (self.current + batchsize) % len(self.examples)
        return np.array(x_batch), np.array(y_batch)

def network(inputs):
#def alexnet_v2(inputs, num_classes=1000,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True,scope='alexnet_v2'):
    num_classes=13
    is_training=True
    dropout_keep_prob=0.5
    spatial_squeeze=True
    scope='alexnet_v2'
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
      # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d], weights_initializer=trunc_normal(0.005),biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, biases_initializer=tf.zeros_initializer, scope='fc8')

      # Convert end_points_collection into a end_point dict.
            #end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            #if spatial_squeeze:
            #    net = tf.squeeze(net, [], name='fc8/squeezed') ### doesn't like
            #    end_points[sc.name + '/fc8'] = net
        net = slim.flatten(net)
        net = slim.fully_connected(net,64, scope = 'fc')
        net = slim.fully_connected(net, 13, activation_fn = None, scope = 'output')
        return net#, end_points
#alexnet_v2.default_image_size = 224
def run_training():

  #Read the training data
  examples, n_classes = read_training_list()
  np.random.seed(42) #shuffle the same way each time for consistency
  np.random.shuffle(examples)
  # TODO TODO - implement some sort of cross validation

  fetcher = Fetcher(examples)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels and mark as input.

    x = tf.placeholder(tf.float32, shape=(None, 512,512,3))
    y_ = tf.placeholder(tf.float32, shape=(None, n_classes)) #n_classes is the number of classes(cells) = 13

    # See "Using instance keys": https://cloud.google.com/ml/docs/how-tos/preparing-models
    # for why we have keys_placeholder
    keys_placeholder = tf.placeholder(tf.int64, shape=(None,))

    # IMPORTANT: Do not change the input map
    inputs = {'key': keys_placeholder.name, 'image': x.name}
    tf.add_to_collection('inputs', json.dumps(inputs))

    # Build a the network
    net = network(x)

    # Add to the Graph the Ops for loss calculation.
    loss = slim.losses.softmax_cross_entropy(net, y_) #####
    tf.scalar_summary(loss.op.name, loss)  # keep track of value for TensorBoard

    # To be able to extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)

    # The prediction will be the index in logits with the highest score.
    # We also use a softmax operation to produce a probability distribution
    # over all possible digits.
    # DO NOT REMOVE OR CHANGE VARIABLE NAMES - used when predicting with a model
    prediction = tf.argmax(net, 1)
    scores = tf.nn.softmax(net)

    # Mark the outputs.
    outputs = {'key': keys.name,
               'prediction': prediction.name,
               'scores': scores.name}
    tf.add_to_collection('outputs', json.dumps(outputs))

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)


    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_output_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      images, labels = fetcher.load_batch(FLAGS.batch_size)
      feed_dict = {x: images, y_: labels}

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 1 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        sys.stdout.flush()
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()


    # Export the model so that it can be loaded and used later for predictions.
    file_io.create_dir(FLAGS.model_dir)
    saver.save(sess, os.path.join(FLAGS.model_dir, 'export'))

    #make world readable for submission to evaluation server
    if FLAGS.model_dir.startswith('gs://'):
        subprocess.call(['gsutil', 'acl','ch','-u','AllUsers:R', FLAGS.model_dir])

    #You probably want to implement some sort of model evaluation here
    #TODO TODO TODO

def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
