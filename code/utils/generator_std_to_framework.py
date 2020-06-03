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
        counter[entity] +=  1

    return counter, indices



def generator(dir, feature_names, output_names, adjecencies_names, interleave_names, training, shuffle=False):
    """
    Parameters
    ----------
    dir:    str
       Name of the entity
    feature_names:    str
       Name of the features to be found in the dataset
    output_names:    str
       Name of the output data to be found in the dataset
    adjecencies_names:    [array]
       CHECK
    interleave_names:    [array]
       First parameter is the name of the interleave, and the second the destination entity
    predict:     bool
        Indicates if we are making predictions, and thus no label is required.
    shuffle:    int (optional)
       Shuffle parameter of the dataset

    """

    dir = dir.decode('ascii')
    feature_names = [x.decode('ascii') for x in feature_names]
    output_names = [o.decode('ascii') for o in output_names]
    adjecencies_names = [[x[0].decode('ascii'),x[1].decode('ascii'),x[2].decode('ascii'), x[3].decode('ascii'), x[4].decode('ascii')] for x in adjecencies_names]
    interleave_names = [[i[0].decode('ascii'), i[1].decode('ascii')] for i in interleave_names]


    samples = glob.glob(str(dir)+'/*.tar.gz')

    if shuffle == 'True':
        random.shuffle(samples)

    for sample_file in samples:
        try:
            tar = tarfile.open(sample_file, 'r:gz')  #read the tar files
            try:
                features = tar.extractfile('data.txt')
            except:
                tf.compat.v1.logging.error('IGNNITION: The file data.txt was not found in ', sample_file )
                sys.exit(1)


            features = json.load(features)
            for sample in features:
                data = {}
                output = []

                #read the features
                for f in feature_names:
                     if f not in sample:
                         raise Exception('A list for feature named ' + str(f) + ' was not found although being expected.')
                     else:
                         data[f] = sample[f]

                #read the output values if we are training
                if training:
                    for name in output_names:
                        if name not in sample:
                            raise Exception('A list for the output named ' + str(name) + ' was not found although being expected.')
                        else:
                            output += sample[name]

                dict = {}

                entities = sample['entities']
                num_nodes, indices = make_indices(entities)

                #create the adjacencies
                for a in adjecencies_names:
                      name, src_entity, dst_entity, ordered, uses_parameters = a

                      if name not in sample:
                         raise Exception('A list for the adjecency vector named ' + name + ' was not found although being expected.')
                      else:
                         adjecency_lists = sample[name]

                         sources_idx = []
                         dest_idx = []
                         seq = []
                         parameters = []

                         # ordered always by destination. (p1: [l1,l2,l3], p2:[l4,l5,l6]...
                         items = adjecency_lists.items()
                         for destination, sources in items:
                             if entities[destination] != dst_entity:
                                 raise Exception('The adjecency list ' + name + ' was expected to be from ' + src_entity + ' to ' + dst_entity +
                                                 ".\n However, " + destination + ' was found which is of type ' + entities[destination] + ' instead of ' + dst_entity)

                             if ordered == 'True':
                                 seq += range(0, len(sources))

                             #check if this adjacency contains extra parameters. This would mean that the sources array would be of shape p0:[[l0,params],[l1,params]...]
                             if isinstance(sources[0],list):
                                 for s in sources:
                                     src_name = s[0]
                                     sources_idx.append((indices[src_name]))
                                     dest_idx.append(indices[destination])

                                     #add the parameters. s[1] should indicate its name
                                     if uses_parameters == 'True':
                                         params = s[1]
                                         parameters.append(params)

                             #in case no extra parameters are provided
                             else:
                                 for s in sources:
                                     if entities[s] != src_entity:
                                         raise Exception('The adjecency list ' + name + ' was expected to be from ' + src_entity + ' to ' + dst_entity +
                                                         ".\n However, " + destination + ' was found which is of type ' + entities[destination] + ' instead of ' + src_entity)

                                     sources_idx.append((indices[s]))
                                     dest_idx.append(indices[destination])


                         data['src_' + name] = sources_idx
                         data['dst_' + name] = dest_idx

                         #add sequence information
                         if ordered == 'True':
                            data['seq_' + src_entity + '_' + dst_entity] = seq
                            dict['seq_' + src_entity + '_' + dst_entity] = seq

                         # remains to check that all adjacencies of the same type have params or not (not a few of them)!!!!!!!!!!
                         if parameters != []:
                             data['params_' + name] = parameters

                #define the graph nodes
                items = num_nodes.items()
                for entity, n_nodes in items:
                     data['num_' + entity] = n_nodes

                #obtain the sequence for the combined message passing. One for each source entity sending to the destination.
                for i in interleave_names:
                    name, dst_entity = i
                    interleave_definition = sample[name]

                    involved_entities = {}

                    total_size = 0
                    n_total = 0
                    counter = 0
                    for entity in interleave_definition:
                        total_size += 1
                        if entity not in involved_entities:
                            involved_entities[entity] = counter

                            seq = dict['seq_' + entity + '_' + dst_entity]
                            n_total += max(seq) + 1
                            counter +=1

                    repetitions = math.ceil(float(n_total) / total_size )   #we exceed the length for sake to make it multiple. Then we will cut it
                    result = [involved_entities[e] for e in interleave_definition]
                    result = np.array((result * repetitions)[:n_total])

                    id = 0
                    for entity in involved_entities:
                        data['indices_' + entity + '_to_' + dst_entity] = np.where(result == id)[0].tolist()
                        id += 1


                if not training:
                    yield data
                else:
                    yield data,output

        except Exception as inf:
            tf.compat.v1.logging.error('IGNNITION: ' + inf)
            sys.exit(1)
