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

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io import file_io


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.')
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
        
    def load_batch(self,batchsize):
        x_batch = []
        y_batch = []
        for i in xrange(batchsize):
            label, files = self.examples[(self.current+i) % len(self.examples)]
            label = label.flatten()
            # If you are getting an error reading the image, you probably have
            # the legacy PIL library installed instead of Pillow
            # You need Pillow
            channels = [ misc.imread(file_io.FileIO(f,'r')) for f in files]
            x_batch.append(np.dstack(channels))
            y_batch.append(label)

        self.current = (self.current + batchsize) % len(self.examples)
        return np.array(x_batch), np.array(y_batch)
        

def network(inputs):
    '''Define the network'''
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = tf.reshape(inputs,[-1, 512,512,3])
        net = slim.conv2d(net, 32, [3,3], scope='conv1')
        net = slim.max_pool2d(net, [4,4], scope = 'conv1')
        net = slim.conv2d(net,64,[3,3], scope = 'conv2')
        net = slim.max_pool2d(net,[4,4], scope = 'pool2')
        net = slim.flatten(net)
        net = slim.fully_connected(net,64, scope = 'fc')
        net = slim.fully_connected(net, 13, activation_fn = None, scope = 'output')
    return net

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
    y_ = tf.placeholder(tf.float32, shape=(None, n_classes))
    
    # See "Using instance keys": https://cloud.google.com/ml/docs/how-tos/preparing-models
    # for why we have keys_placeholder
    keys_placeholder = tf.placeholder(tf.int64, shape=(None,))

    # IMPORTANT: Do not change the input map
    inputs = {'key': keys_placeholder.name, 'image': x.name}
    tf.add_to_collection('inputs', json.dumps(inputs))

    # Build a the network
    net = network(x)

    # Add to the Graph the Ops for loss calculation.
    loss = slim.losses.softmax_cross_entropy(net, y_)
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

    # Start the training loop.
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
