'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

# -*- coding: utf-8 -*-


import sys
import configparser
import tensorflow as tf
import datetime
import warnings
from generate_model import *
from json_operations import *
import glob
import tarfile
import json
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG._interpolation = configparser.ExtendedInterpolation()
CONFIG.read('./train_options.ini')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
warnings.filterwarnings("ignore")


def create_model():
    json_path = CONFIG['PATHS']['json_path']
    dimensions = find_dataset_dimensions(CONFIG['PATHS']['train_dataset'])
    model_info = Model_information(json_path, dimensions)  # read json

    return model_info


def find_dataset_dimensions(path):
    sample = glob.glob(str(path) + '/*.tar.gz')[0]  #choose one single file to extract the dimensions
    try:
        tar = tarfile.open(sample, 'r:gz')  # read the tar files
    except:
        tf.compat.v1.logging.error('IGNNITION: The file data.json was not found in ' + sample)
        sys.exit(1)

    try:
        file_samples = tar.extractfile('data.json')
        sample_data = json.load(file_samples)[0]  # one single sample
        dimensions = {}
        for k, v in sample_data.items():
            if not isinstance(v,dict):  #if it's a feature
                if isinstance(v,list) and isinstance(v[0], list):
                    dimensions[k] = len(v[0])
                else:
                    dimensions[k] = 1

            elif v:   #if its either the entity or an adjacency (it is a dictionary, that is non-empty)
                first_key = list(v.keys())[0]
                element = v[first_key]
                if (not isinstance(element[0], str)) and isinstance (element[0], list):
                    dimensions[k] = len(element[0][1])  #the element[0][1] is the adjacency node. The second is the edge information
                else:
                    dimensions[k] = 0
        return dimensions

    except:
        tf.compat.v1.logging.error('IGNNITION: Failed to read the data file ' + sample)
        sys.exit(1)


def str_to_bool(a):
    if a == 'True':
        return True
    else:
        return False


def train_and_evaluate(model):
    """
    Parameters
    ----------
    model:    object
        Object with the json information model
    """
    print()
    tf.compat.v1.logging.warn('IGNNITION: Starting the training and evaluation process...\n---------------------------------------------------------------------------\n')
    set_model_info(model)

    filenames_train = CONFIG['PATHS']['train_dataset']
    filenames_eval = CONFIG['PATHS']['eval_dataset']

    model_dir = CONFIG['PATHS']['model_dir']  #checkpoint path
    model_dir = model_dir + '/experiment_' + str(datetime.datetime.now())

    if CONFIG.has_option('PATHS', 'warm_start_path'):
        warm_start_setting = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=CONFIG['PATHS']['warm_start_path'],
            vars_to_warm_start=["kernel.*", "recurrent_kernel.*", "bias.*"])

    else:
        warm_start_setting = None


    device = 'CPU'
    if CONFIG.has_option('TRAINING_OPTIONS', 'execute_gpu'):
        if CONFIG['TRAINING_OPTIONS']['execute_gpu'] == 'True':
            device = 'GPU'
        else:
            device = 'CPU'

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=int(CONFIG['TRAINING_OPTIONS']['save_checkpoints_secs']),
        keep_checkpoint_max=int(CONFIG['TRAINING_OPTIONS']['keep_checkpoint_max']),
        session_config=tf.compat.v1.ConfigProto(device_count={device: 0})
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        warm_start_from = warm_start_setting,
        config=my_checkpointing_config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: tfrecord_input_fn(filenames_train,shuffle = True),
        max_steps=int(CONFIG['TRAINING_OPTIONS']['train_steps']))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: tfrecord_input_fn(filenames_eval,shuffle = str_to_bool(CONFIG['TRAINING_OPTIONS']['shuffle_eval_samples'])), throttle_secs=int(CONFIG['TRAINING_OPTIONS']['throttle_secs']), steps=int(CONFIG['TRAINING_OPTIONS']['eval_samples']))


    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def predict(model_info):
      """
      Parameters
      ----------
      model_info:    object
       Object with the json information model
      """

      print()
      tf.compat.v1.logging.warn('IGNNITION: Starting to make the predictions...\n---------------------------------------------------------\n')
      set_model_info(model_info)

      graph = tf.Graph()
      tf.compat.v1.disable_eager_execution()

      try:
          warm_path = CONFIG['PATHS']['warm_start_path']
      except:
          tf.compat.v1.logging.error('IGNNITION: The path of the model to use for the predictions is unspecified. Please add a field warm_start_path in the train_options.ini with the corresponding path to the model you want to restore.')
          sys.exit(0)

      try:
          data_path = CONFIG['PATHS']['predict_dataset']
      except:
          tf.compat.v1.logging.error('IGNNITION: The path of dataset to use for the prediction is unspecified. Please add a field predict_dataset in the train_config.ini file with the corresponding path to the dataset you want to predict.')
          sys.exit(0)


      with graph.as_default():
          model = ComnetModel()

          it = tfrecord_input_fn(data_path, training=False)  #this should not return a label since we might not have one!!!!!!!!!!!!
          features = it.get_next()
          predictions = model(features, training=False)  #this predictions still need to be denormalized (or to do the predictions with the estimators)

          #automatic denormalization
          output_names, _, output_denormalizations = model_info.get_output_info()  # for now suppose we only have one output type
          try:
              pred = eval(output_denormalizations[0])(predictions, output_names[0])
          except:
              tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_names[
                  0] + ' was not defined. The output will be normalized.')


      with tf.compat.v1.Session(graph=graph) as sess:
          sess.run(tf.compat.v1.local_variables_initializer())
          sess.run(tf.compat.v1.global_variables_initializer())
          saver = tf.compat.v1.train.Saver()

          # path to the checkpoint we want to restore
          saver.restore(sess, warm_path)

          all_predictions = []
          try:
              sess.run(it.initializer)
              while True:
                  p = sess.run([pred])
                  p = np.array(p)
                  p = p.flatten()
                  print(p)
                  all_predictions.append(p)

          except tf.errors.OutOfRangeError:
              pass

          return all_predictions



def debug(model_description):
    """
    Parameters
    ----------
    model_description:    object
        Object with the json information model
    """

    print()
    tf.compat.v1.logging.warn('IGNNITION: Generating the debug model... \n---------------------------------------------------------\n')
    set_model_info(model_description)

    filenames_train = CONFIG['PATHS']['train_dataset']
    graph = tf.Graph()
    tf.compat.v1.disable_eager_execution()

    with graph.as_default():
        model = ComnetModel()
        it = tfrecord_input_fn(filenames_train, training = False)
        features = it.get_next()
        predictions = model(features, training=False)


    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(it.initializer)

    tf.compat.v1.summary.FileWriter('../debug_model/', graph=sess.graph)
    tf.compat.v1.logging.warn('IGNNITION: The debug model has been generated.')

