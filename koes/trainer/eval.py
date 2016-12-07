#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
# Modified by dkoes.

r"""A script evaluating a saved session on new TEST data.

  # Using a model from the local filesystem:
  python eval.py --model_dir=output/${JOB_NAME}/model TEST

"""

import argparse
import collections
import json
import os
import time
import subprocess
import numpy as np
import sys
from scipy import misc
from google.cloud.ml import session_bundle
from tensorflow.python.lib.io import file_io
from sklearn import metrics

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

def parse_args():
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Evaluate predictions.')

  argparser.add_argument(
      '--test_dir',
      dest='test_dir',
      help=('Directory with TEST datafile and images'))

  argparser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))


  return argparser.parse_args()


def read_test_list(testdir):
    """
    Read <train_data_dir>/TEST which containing paths and labels in
    the format label, channel1 file, channel2 file, channel3 
    Returns:
        List with all filenames in file image_list_file
    """
    image_list_file = testdir + '/TEST'
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
        filenames.append([ testdir + '/' + c for c in (c1,c2,c3)])
    
    return zip( labels,filenames)

if __name__ == '__main__':
  args = parse_args()


  session, _ = session_bundle.load_session_bundle_from_path(args.model_dir)
  # get the mappings between aliases and tensor names
  # for both inputs and outputs
  input_alias_map = json.loads(session.graph.get_collection('inputs')[0])
  output_alias_map = json.loads(session.graph.get_collection('outputs')[0])
  aliases, tensor_names = zip(*output_alias_map.items())
  examples = read_test_list(args.test_dir)
  start_time = time.time() 
  y_true = []
  y_pred = []
  for (label, files) in examples:
      channels = [ misc.imread(file_io.FileIO(f,'r')) for f in files]
      image = np.dstack(channels)

      feed_dict = {input_alias_map['image']: [image]}
      predict, scores = session.run(fetches=[output_alias_map['prediction'],output_alias_map['scores']], feed_dict=feed_dict)
      y_true.append(np.argmax(label))
      y_pred.append(predict[0])
  
  duration = time.time()-start_time
  accuracy = metrics.accuracy_score(y_true,y_pred)
  f1macro = metrics.f1_score(y_true,y_pred,average='macro')
  f1micro = metrics.f1_score(y_true,y_pred,average='micro')

  print "PredictTime",duration
  print "Accuracy",accuracy
  print "F1macro",f1macro
  print "F1micro",f1micro
