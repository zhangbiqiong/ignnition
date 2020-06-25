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

    get_output_models(self)
        Returns the readout architectures of the model

    get_output_info(self)
        ?

    get_all_features(self)
        Returns all the features defined in the model, no matter the entity they are assigned to

    get_feature_size(self, feature)
        Returns the size of a feature

    get_adjecency_info(self)

    set_entity(self, name, hidden_state_dimension)
        Sets an entity to have a certain hidden-state dimension

    set_feature(self, name, size=1, normalization = None)
        Sets a feature to a certain entity

    set_num_iterations(self, it)
        Sets a number of iterations for the mp phase

    create_mp_step(self,name)
        Creates a message passing step

    create_message_passing(self, type, source, destination, adj_vector)
        Creates a message passing

    add_combined_mp(self, step, destination_entity, message_combination, update)
        Create a combined message passing

    create_readout(self, type, entity, output_label, output_normalization=None, output_denormalization = None)
        Creates a readout model

    set_loss(self, l)
        Sets the loss of the model

    set_optimizer(self, **o)
        Sets the optimizer of the model

    set_schedule(self, **dict)
        Sets the schedule of the model
    """

    def __init__(self, path = None):
        """
        Parameters
        ----------
        path:    str (optional)
            Path of the json file with the model description
        """

        #in case JSON is used
        if path != None:
            data = self.__read_json(path)
            validate(instance=data, schema=self.__read_json('./utils/schema.json')) #validate that the json is well defined

            self.nn_architectures = self.__obtain_feed_forward_mapping(data['feed_forward_models'])
            self.entities = self.__obtain_entities(data['entities'])
            self.iterations_mp = data['mp_phase']['num_iterations']
            self.mp_instances = self.__obtain_mp_instances(data['mp_phase']['architecture'])
            self.combined_mp_options = self.__calculate_mp_combination_options(data)    #if there is any
            self.outputs = self.__obtain_output_models(data['output'])
            self.training_options = self.__obtain_training_options(data)
            self.entities_dimensions = self.__get_entities_dimensions()

        #in case API is used
        else:
          self.entities = []
          self.iterations_mp = 8  #default
          self.mp_instances = []
          self.combined_mp_options = {}
          self.outputs = []
          self.training_options = {}
          self.entities_dimensions = {}


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



    def __obtain_feed_forward_mapping(self, models):
        result = {}

        for m in models:
            result[m['model_name']] = m['model_architecture']

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
        for op in m['message']:
            if op['type'] == 'apply_nn':
                architecture = copy.deepcopy(self.nn_architectures[op['nn_name']])
                del op['nn_name']
                op['architecture'] = architecture

        #add the update nn architecture
        if 'update' in m and m['update']['type'] == 'apply_nn':
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
            aux = [Message_Passing(self.__add_nn_architecture(m)) for m in step['message_passings']]
            mp_instances.append([step['step_name'],aux])

        return mp_instances


    def __calculate_mp_combination_options(self, data):
        """
        Parameters
        ----------
        data:    dict
           Dictionary with the definition of each combined message passing
        """

        m = data["mp_phase"]
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


    def __obtain_output_models(self, outputs):
        """
        Parameters
        ----------
        outputs:    dict
           Dictionary with the definition of each output model
        """

        result = []
        for output in outputs:
            r = Readout_model(self.__add_readout_architecture(output))
            result.append(r)
        return result


    def __obtain_training_options(self, data):
        """
        Parameters
        ----------
        data:    dict
           Dictionary with the definition of the training options of the framework
        """

        train_hp = data['training_hyperparameters']
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


    def get_output_models(self):
        return self.outputs

    def get_output_info(self):
        result_names = [o.output_label for o in self.outputs]
        result_norm = [o.output_normalization for o in self.outputs]
        result_denorm = [o.output_denormalization for o in self.outputs]
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


    #----------------------------------------------------------------

    #PUBLIC SETTERS (API)
    def set_entity(self, name, hidden_state_dimension):
        """
        Parameters
        ----------
        name:    str
           Name of the entity
        hidden_state_dimension:    int
           Dimension of the hidden states of this entity nodes
        """

        e = Entity({"name":name,
                    "hidden_state_dimension": hidden_state_dimension})
        self.entities.append(e)
        self.entities_dimensions[name] = hidden_state_dimension

    def set_feature(self, name, size=1, normalization = None):
        """
        Parameters
        ----------
        name:    str
           Name of the entity
        size:    int (optional)
           Dimension of the feature
        size:    str (optional)
           Type of normalization to be applied to this feature
        """

        if normalization == None:
            normalization = "None"

        f = Feature({"name":name,
                     "size": size,
                     "normalization": normalization})
        self.entities[-1].add_feature(f)    #add the feature to the last added entity

    def set_num_iterations(self, it):
        """
        Parameters
        ----------
        it:    int
           Number of iterations of the algorithm
        """

        self.iterations_mp = it

    def create_mp_step(self,name):
        """
        Parameters
        ----------
        name:    str
           Name of the message passing step
        """

        self.mp_instances.append([name, []])

    def create_message_passing(self, type, source, destination, adj_vector):
        """
        Parameters
        ----------
        type:    str
           Type of message passing
        source:    int
           Source entity of the message passing
        destination: string
            Destination entity of the message passing
        adj_vector: sting
            Name of the key in the dataset with the adjacency list from source entity to destination entity
        """

        m = Message_Passing({
            "type":type,
            "destination_entity": destination,
            "source_entity": source,
            "adj_vector": adj_vector
        })
        self.mp_instances[-1][1].append(m)
        return m


    def add_combined_mp(self, step, destination_entity, message_combination, update):
        """
        Parameters
        ----------
        step:    str
            Step of the algorithm in which this combined message passing is located
        destination_entity: str
            Destination of the combined message passing
        message_combination:    str
            Type of combination of the messages from different source entities
        update:     str
            Update type that the destination nodes shall use
        """

        if step not in self.combined_mp_options:
            self.combined_mp_options[step] = []

        self.combined_mp_options[step].append({
            "destination_entity":destination_entity,
            "message_combination": message_combination,
            "update": update,
        })


    def create_readout(self, type, entity, output_label, output_normalization=None, output_denormalization = None):
        """
        Parameters
        ----------
        type:    str
            Type of readout (local or global)
        entity:     str
            Entity which hidden states shall be used for the predicitons
        output_label:   str
            Key from the dataset where the data of labels can be found
        output_normalization:   str
            Strategy to normalize the labels and the predictions. Should match with a functioned defined in main.py.
        output_denormalization:     str
            Strategy to denormalize the labels and the predictions. Should match with a functioned defined in main.py.
        """

        if output_normalization == None:
            output_normalization = "None"

        r = Readout_model({
            "type":type,
            "entity": entity,
            "output_label": output_label,
            "output_normalization": output_normalization,
            "output_denormalization": output_denormalization
        })
        self.outputs.append(r)
        return r


    def set_loss(self, l):
        """
        Parameters
        ----------
        dict:    dict
           Dictionary with the parameters to create the loss model form keras.
        """

        self.training_options['loss'] = l

    def set_optimizer(self, **o):
        """
        Parameters
        ----------
        dict:    dict
           Dictionary with the parameters to create the optimizer model form keras.
        """

        self.training_options['optimizer'] = o


    def set_schedule(self, **dict):
        """
        Parameters
        ----------
        dict:    dict
           Dictionary with the parameters to create the schedule model form keras.
        """

        self.training_options['schedule'] = dict

