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


import glob
import json
import sys
import tarfile
import numpy as np
import math
import random
import tensorflow as tf


def make_indices(entities):
    """
    Parameters
    ----------
    entities:    dict
       Dictionary with the information of each entity
    """

    counter = {}
    indices = {}
    items = entities.items()
    for node, entity in items:
        if entity not in counter:
            counter[entity] = 0

        indices[node] = counter[entity]
        counter[entity] += 1

    return counter, indices


def generator(dir, feature_names, output_name, adj_names, interleave_names, additional_input, training,
              shuffle=False):
    """
    Parameters
    ----------
    dir:    str
       Name of the entity
    feature_names:    str
       Name of the features to be found in the dataset
    output_names:    str
       Name of the output data to be found in the dataset
    adj_names:    [array]
       CHECK
    interleave_names:    [array]
       First parameter is the name of the interleave, and the second the destination entity
    predict:     bool
        Indicates if we are making predictions, and thus no label is required.
    shuffle:    bool
       Shuffle parameter of the dataset

    """

    dir = dir.decode('ascii')
    feature_names = [x.decode('ascii') for x in feature_names]
    output_name = output_name.decode('ascii')
    adj_names = [[x[0].decode('ascii'), x[1].decode('ascii'), x[2].decode('ascii'), x[3].decode('ascii')] for x in adj_names]
    interleave_names = [[i[0].decode('ascii'), i[1].decode('ascii')] for i in interleave_names]
    additional_input = [x.decode('ascii') for x in additional_input]
    samples = glob.glob(str(dir) + '/*.tar.gz')

    if shuffle == True:
        random.shuffle(samples)

    for sample_file in samples:
        try:
            tar = tarfile.open(sample_file, 'r:gz')  # read the tar files
            try:
                file_samples = tar.extractfile('data.json')

            except:
                tf.compat.v1.logging.error('IGNNITION: The file data.json was not found in ', sample_file)
                sys.exit(1)

            file_samples = json.load(file_samples)
            for sample in file_samples:
                data = {}
                output = []

                # read the features
                for f in feature_names:
                    if f not in sample:
                        raise Exception(
                            'A list for feature named "' + str(f) + '" was not found although being expected.')
                    else:
                        data[f] = sample[f]

                # read additional input name
                for a in additional_input:
                    if a not in sample:
                        raise Exception('The input name "' + str(a) + '" was not found although being expected.')
                    else:
                        data[a] = sample[a]

                # read the output values if we are training
                if training:
                    if output_name not in sample:
                        raise Exception('A list for the output named "' + str(
                            output_name) + '" was not found although being expected.')
                    else:
                        value = sample[output_name]
                        if not isinstance(value, list):
                            value = [value]

                        output += value

                dict = {}

                entities = sample['entities']
                num_nodes, indices = make_indices(entities)

                # create the adjacencies
                for a in adj_names:
                    name, src_entity, dst_entity, uses_parameters = a

                    if name not in sample:
                        raise Exception(
                            'A list for the adjecency vector named "' + name + '" was not found although being expected.')
                    else:
                        adjecency_lists = sample[name]
                        src_idx, dst_idx, seq, parameters = [], [], [], []

                        # ordered always by destination. (p1: [l1,l2,l3], p2:[l4,l5,l6]...
                        items = adjecency_lists.items()
                        for destination, sources in items:
                            if entities[destination] != dst_entity:
                                raise Exception(
                                    'The adjecency list "' + name + '" was expected to be from ' + src_entity + ' to ' + dst_entity +
                                    '.\n However, "' + destination + '" was found which is of type "' + entities[
                                        destination] + '" instead of ' + dst_entity)

                            seq += range(0, len(sources))

                            # check if this adjacency contains extra parameters. This would mean that the sources array would be of shape p0:[[l0,params],[l1,params]...]
                            if isinstance(sources[0], list):
                                for s in sources:
                                    src_name = s[0]
                                    src_idx.append((indices[src_name]))
                                    dst_idx.append(indices[destination])

                                    # add the parameters. s[1] should indicate its name
                                    if uses_parameters == 'True':   parameters.append(s[1])

                            # in case no extra parameters are provided
                            else:
                                for s in sources:
                                    if entities[s] != src_entity:
                                        raise Exception(
                                            'The adjecency list "' + name + '" was expected to be from "' + src_entity + '" to "' + dst_entity +
                                            '.\n However, "' + destination + '" was found which is of type "' +
                                            entities[destination] + '" instead of "' + src_entity)

                                    src_idx.append((indices[s]))
                                    dst_idx.append(indices[destination])

                        data['src_' + name] = src_idx
                        data['dst_' + name] = dst_idx

                        # add sequence information
                        data['seq_' + src_entity + '_' + dst_entity] = seq
                        dict['seq_' + src_entity + '_' + dst_entity] = seq

                        # remains to check that all adjacencies of the same type have params or not (not a few of them)!!!!!!!!!!
                        if parameters != []:    data['params_' + name] = parameters

                # define the graph nodes
                items = num_nodes.items()
                for entity, n_nodes in items:
                    data['num_' + entity] = n_nodes

                # obtain the sequence for the combined message passing. One for each source entity sending to the destination.
                for i in interleave_names:
                    name, dst_entity = i
                    interleave_definition = sample[name]

                    involved_entities = {}
                    total_sequence = []
                    total_size, n_total, counter = 0, 0, 0

                    for entity in interleave_definition:
                        total_size += 1
                        if entity not in involved_entities:
                            involved_entities[entity] = counter  # each entity a different value (identifier)

                            seq = dict['seq_' + entity + '_' + dst_entity]
                            n_total += max(seq) + 1  # superior limit of the size of any destination
                            counter += 1

                        # obtain all the original definition in a numeric format
                        total_sequence.append(involved_entities[entity])

                        # we exceed the length for sake to make it multiple. Then we will cut it
                    repetitions = math.ceil(float(n_total) / total_size)
                    result = np.array((total_sequence * repetitions)[:n_total])

                    for entity in involved_entities:
                        id = involved_entities[entity]
                        data['indices_' + entity + '_to_' + dst_entity] = np.where(result == id)[0].tolist()

                if not training:
                    yield data
                else:
                    yield data, output

        except KeyboardInterrupt:
            sys.exit(1)

        except Exception as inf:
            tf.compat.v1.logging.error('IGNNITION: ' + str(inf))
