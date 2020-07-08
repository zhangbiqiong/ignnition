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


import json
from jsonschema import validate
from auxilary_classes import *
import copy
import sys

class Model_information:
    """
    Attributes
    ----------
    entities:   dict
        Contains the different entity object

    iterations_mp:  int
        Number of iterations that the mp phase will have

    mp_instances:   dict
        Contains the different MP object

    combined_mp_options:    dict
        Information of the different combined message passings

    outputs:    dict
        Readout architectures to be used for prediction

    training_options:   dict
        Training options and parameters

    entities_dimensions:    dict
        Maps each entity with its hidden state dimension.


    Methods:
    ----------
    __read_json(self, path)
        Reads the json file from the path and returns the data as a dictionary

    __obtain_entities(self, entities)
        Creates the entities specified in the entities dictionary

    __obtain_mp_instances(self, inst)
        Creates the message passing object

    __calculate_mp_combination_options(self, data)
        Creates the combined message passing object.

    __obtain_output_models(self, outputs)
        Creates the different readout models

    __obtain_training_options(self, data)
        Specifies the training options of the model

    __get_entities_dimensions(self)
        Obtains the dictionary that maps entities and its hidden-state dimensions

    get_entities_dimensions(self)
        Returns the dictionary that maps entities and its hidden-states dimensions

    get_entities(self)
        Returns the entities of the model

    get_combined_mp_options(self)
        Returns the combined message passings

    get_combined_mp_sources(self, dst_entity, step_name)
        Returns the sources that are part of a combined mp in step_name sending to dst_entity

    get_combined_sources(self)
        Return all sources that are part of a combined mp

    get_mp_iterations(self)
        Returns the number of iterations of the mp phase

    get_interleave_tensors(self)
        ?

    get_mp_instances(self)
        Returns the message passing objects of the model

    get_schedule(self)
        Returns the schedule set in the model

    get_optimizer(self)
        Returns the optimizer set in the model

    get_loss(self)
        Returns the loss set in the model

    get_output_operations(self)
        Returns the readout architectur of the model in the form of operations

    get_output_info(self)
        ?

    get_all_features(self)
        Returns all the features defined in the model, no matter the entity they are assigned to

    get_feature_size(self, feature)
        Returns the size of a feature

    get_adjecency_info(self)
    """

    def __init__(self, path, dimensions):
        """
        Parameters
        ----------
        path:    str (optional)
            Path of the json file with the model description
        """

        #read and validate the json file
        data = self.__read_json(path)
        validate(instance=data,schema=self.__read_json('./utils/schema.json'))  # validate that the json is well defined
        self.__validate_model_description(data)
        self.__add_dimensions(data, dimensions) #add the dimension of the features and of the edges

        self.nn_architectures = self.__obtain_feed_forward_mapping(data['neural_networks'])
        self.entities = self.__obtain_entities(data['entities'])
        self.iterations_mp = data['message_passing']['num_iterations']
        self.mp_instances = self.__obtain_mp_instances(data['message_passing']['architecture'])
        self.combined_mp_options = self.__calculate_mp_combination_options(data)    #if there is any
        self.readout_operations = self.__obtain_readout_operations(data['readout'])
        self.training_options = self.__obtain_training_options(data)
        self.entities_dimensions = self.__get_entities_dimensions()



    #PRIVATE
    def __read_json(self, path):
        """
        Parameters
        ----------
        path:    str (optional)
            Path of the json file with the model description
        """

        with open(path) as json_file:
            data = json.load(json_file)
            return data


    def __add_dimensions(self, data, dimensions):
        for e in data['entities']:
            for f in e['features']:
                name = f['name']
                e['size'] = dimensions[name]    #add the dimension of the feature

        for s in data['message_passing']['architecture']:
            aux = s['mp_step']
            for a in aux:
                name = a['adj_vector']
                a['extra_parameters'] = dimensions[name]



    #validate that all the nn_name are correct. Validate that all source and destination entities are correct. Validate that all the inputs in the message function are correct
    def __validate_model_description(self, data):
        steps = data['message_passing']['architecture']

        sources = []
        destinations = []
        nn_name_called = []
        output_names = ['hs_source', 'hs_dest', 'edge_params']
        input_names = []
        for s in steps:
            s = s['mp_step']
            for m in s: #for every message-passing
                sources.append(m['source_entity'])
                destinations.append(m['destination_entity'])

                if 'message' in m:
                    for op in m['message']: #for every operation
                        if op['type'] == 'neural_network':
                            nn_name_called.append(op['nn_name'])
                            input_names += op['input']

                            if 'output_name' in op:
                                output_names.append(op['output_name'])

        readouts = data['readout']
        for r in readouts:
            nn_name_called.append(r['nn_name'])


        #now check the entities
        entity_names = [a['name'] for a in data['entities']]
        nn_names = [a['nn_name'] for a in data['neural_networks']]
        try:

            for a in sources:
                if a not in entity_names:
                    raise Exception('The source entity ' + a + ' was used in a message passing. However, there is no such entity. \n Please check the spelling or define a new entity.')

            for d in destinations:
                if d not in entity_names:
                    raise Exception('The destination entity ' + d + ' was used in a message passing. However, there is no such entity. \n Please check the spelling or define a new entity.')


            #check the nn_names
            for name in nn_name_called:
                if name not in nn_names:
                    raise Exception('The name ' + name + " is used as a reference to a neural network (nn_name), even though the neural network was not defined. \n Please make sure the name is correctly spelled or define a neural network named " + name)

            #check the output and input names
            for i in input_names:
                if i not in output_names:
                    raise Exception('The name ' + i + " was used as input of a message creation operation even though it wasn't the output of one.")

        except Exception as inf:
            tf.compat.v1.logging.error('IGNNITION: ' + str(inf) + '\n')
            sys.exit(1)




    def __obtain_feed_forward_mapping(self, models):
        result = {}

        for m in models:
            result[m['nn_name']] = m['nn_architecture']

        return result


    def __obtain_entities(self, entities):
        """
        Parameters
        ----------
        entities:    dict
           Dictionary with the definition of each entity
        """

        l = [Entity(e) for e in entities]
        return l


    #substitutes the referenced name by the correct architecture
    def __add_nn_architecture(self, m):

        #we need to find out what the input dimension is

        #add the message_creation nn architecture
        if 'message' in m:
            for op in m['message']:
                if op['type'] == 'neural_network':
                    architecture = copy.deepcopy(self.nn_architectures[op['nn_name']])
                    del op['nn_name']
                    op['architecture'] = architecture

        #add the update nn architecture
        if 'update' in m and m['update']['type'] == 'neural_network':
            architecture = copy.deepcopy(self.nn_architectures[m['update']['nn_name']])
            del m['update']['nn_name']
            m['update']['architecture'] = architecture

        return m


    def __obtain_mp_instances(self, inst):
        """
        Parameters
        ----------
        inst:    dict
           Dictionary with the definition of each message passing
        """
        mp_instances = []

        for step in inst:
            aux = [Message_Passing(self.__add_nn_architecture(m)) for m in step['mp_step']]
            mp_instances.append([step['step_name'],aux])

        return mp_instances


    def __calculate_mp_combination_options(self, data):
        """
        Parameters
        ----------
        data:    dict
           Dictionary with the definition of each combined message passing
        """

        m = data["message_passing"]
        if "combined_message_passing_options" in m.keys():
            aux = m["combined_message_passing_options"]
            dict = {}
            for mp_info in aux:
                step_name = mp_info['step']
                if step_name not in dict:
                    dict[step_name] = []
                c = Combined_mp(mp_info)
                dict[step_name].append(c)
            return dict
        else:
            return {}


    def __add_readout_architecture(self, output):
        name = output['nn_name']
        architecture = copy.deepcopy(self.nn_architectures[name])
        del output['nn_name']
        output['architecture'] = architecture

        return output


    def __obtain_readout_operations(self, output_operations):
        """
        Parameters
        ----------
        output_operations:    dict
           List of dictionaries with the definition of the operations forming one readout model
        """
        result = []
        for op in output_operations:
            if op['type'] == 'predict':
                r = Predicting_operation(self.__add_readout_architecture(op))
                result.append(r)

            elif op['type'] == 'pooling':
                r = Pooling_operation(op)
                result.append(r)

        return result


    def __obtain_training_options(self, data):
        """
        Parameters
        ----------
        data:    dict
           Dictionary with the definition of the training options of the framework
        """

        train_hp = data['learning_options']
        dict = {}

        dict['loss'] = train_hp['loss'] #required
        dict['optimizer'] = train_hp['optimizer']   #required

        if 'schedule' in train_hp:
            dict['schedule'] = train_hp['schedule'] #optional
        return dict


    def __get_entities_dimensions(self):
        dict = {}
        for entity in self.entities:
            dict[entity.name] = entity.hidden_state_dimension
        return dict

    # ----------------------------------------------------------------
    #PUBLIC FUNCTIONS GETTERS
    def get_entities_dimensions(self):
        return self.entities_dimensions

    def get_entities(self):
        return self.entities

    def get_combined_mp_options(self):
        return self.combined_mp_options

    def get_combined_mp_sources(self, dst_entity, step_name):
        sources = []
        for step in self.mp_instances:
            if step[0] == step_name:   #check if we are in fact within the step we care about
                for m in step[1]:  # this is just one value
                    if (m.type == "multi_source") & (m.destination_entity== dst_entity):
                        sources.append(m.source_entity)

        return sources

    def get_combined_sources(self):
        result = []
        for step in self.mp_instances:
            for m in step[1]:
                if m.type == "multi_source" and m.aggregation == 'combination':
                    result.append([m.source_entity,m.destination_entity])
        return result

    def get_mp_iterations(self):
       return self.iterations_mp


    def get_interleave_tensors(self):
        if self.combined_mp_options != {}:
            combination_blocks = list(self.combined_mp_options.values())[0]
            result = [[combined_message.combination_definition, combined_message.destination_entity] for combined_message in combination_blocks if combined_message.message_combination == 'interleave' ]
            return result
        else:
            return []

    def get_mp_instances(self):
        return self.mp_instances

    def get_schedule(self):
        return self.training_options['schedule']

    def get_optimizer(self):
        return self.training_options['optimizer']

    def get_loss(self):
        return self.training_options['loss']


    def get_readout_operations(self):
        return self.readout_operations

    def get_output_info(self):
        result_names = [o.label for o in self.readout_operations if o.type=='predict']
        result_norm = [o.label_normalization for o in self.readout_operations if o.type == 'predict']
        result_denorm = [o.label_denormalization for o in self.readout_operations if o.type == 'predict']
        return result_names, result_norm, result_denorm

    def get_all_features(self):
        all_features = []
        for e in self.entities:
            all_features += e.features

        return all_features

    def get_feature_size(self, feature):
        if 'size' in feature:
            return feature['size']
        else:
            return 1

    def get_adjecency_info(self):
        result = []
        for step in self.mp_instances:
            for instance in step[1]:
                result.append(instance.get_instance_info())

        return result
