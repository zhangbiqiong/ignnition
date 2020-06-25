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


import tensorflow as tf

class Feature:
    """
    Attributes
    ----------
    name:    str
        Name of the feature
    size:   int
        Dimension of the feature
    normalization:  str
        Type of normalization to be applied to this feature

    """

    def __init__(self, f):
        """
        Parameters
        ----------
        f:    dict
            Dictionary with the required attributes
        """

        self.name = f['name']
        self.size = 1
        self.normalization = 'None'

        if 'size' in f:
            self.size = f['size']

        if 'normalization' in f:
            self.normalization = f['normalization']


class Entity:
    """
    Attributes
    ----------
    name:    str
        Name of the feature
    hidden_state_dimension:   int
        Dimension of the hiddens state of these entity node's
    features:  array
        Array with all the feature objects that it contains


    Methods:
    ----------
    get_entity_total_feature_size(self)
        Sum of the dimension of all the features contained in this entity

    get_features_names(self)
        Return all the names of the features of this entity

    add_feature(self, f)
        Adds a feature to this entity

    """

    def __init__(self, dict):
        """
        Parameters
        ----------
        dict:    dict
            Dictionary with the required attributes
        """

        self.name = dict['name']
        self.hidden_state_dimension = dict['hidden_state_dimension']
        self.features = []

        if 'features' in dict:
            for f in dict['features']:
                aux = Feature(f)
                self.features.append(aux)

    def get_entity_total_feature_size(self):

        total = 0
        for f in self.features:
            total += f.get_size()

        return total

    def get_features_names(self):
        result = []
        for f in self.features:
            result.append(f.get_name())

        return result

    def add_feature(self, f):
        self.features.append(f)


class Operation:
    def __init__(self, type):
        self.type = type    #type of operation

class Apply_nn(Operation):
    def __init__(self, op):
        super(Apply_nn, self).__init__(type="feed_forward_nn")
        self.input = op['input']

        if 'output_name' in op:
            self.output_name = op['output_name']

        else:
            self.output_name = 'None'

        #we need somehow to find the number of extra_parameters beforehand
        self.model = Feed_forward_message_creation(op['architecture'], 0)


class Apply_rnn(Operation):
    def __init__(self, op):
        super(Apply_rnn, self).__init__(type="recurrent_nn")

        #Here initialize the recursive neural network (op might contain any additional parameter of the RNN?
        del op['type']
        recurrent_type = op['recurrent_type']
        del op['recurrent_type']
        self.model = Recurrent_Cell(recurrent_type, op)


class Combined_mp:
    """
    Attributes
    ----------
    destination_entity:     str
        Name of the destination entity of this combined message passing
    message_combination:    str
        Type of message combination
    combination_definition:    str (optional)
        Type of combination strategy to be used to combine messages from different type of nodes
    update:    object
        Model to be used for the update
    """

    def __init__(self, dict):
        """
        Parameters
        ----------
        dict:    dict
            Dictionary with the required attributes
        """

        self.destination_entity = dict['destination_entity']
        self.message_combination = dict['message_combination']
        if 'combination_definition' in dict:
            self.combination_definition = dict['combination_definition']
        else:
            self.combination_definition = None

        params = dict['update']
        type = params['recurrent_type']
        del params['recurrent_type']
        self.update = Recurrent_Cell(type, params)



class Message_Passing:
    """
    Attributes
    ----------
    type:   str
        Type of message passing (individual or combined)
    destination_entity:     str
        Name of the destination entity
    source_entity:      str
        Name of the source entity
    adj_vector:     str
        Name from the dataset where the adjacency list can be found
    extra_parameters:    int (optional)
        Number of extra parameters to be used
    aggregation:     str
        Type of aggregation strategy
    update:     object
        Object with the update model to be used
    update_type:     str
        Indicates if it uses feed-forward or recurrent update
    formation_type:     str
        Indicates if it used feed-forward or recurrent update
    message_formation:      object (optional)
        Feed forward model for the message formation


    Methods:
    --------
    create_update(self,dict)
        Create a model to update the hidden state of the destination entity

    find_type_of_message_creation(self, type)


    create_message_formation(self, dict)
        Creates the model to be used for the creation of the messages of the source entity

    get_instance_info(self):
        Returns an array with all the relevant information of this message passing

    set_message_formation(self, message_neural_net, number_extra_parameters = 0)
        Sets the message formation strategy by creating the appropriate model to do so.

    add_message_formation_layer(self, **dict):
        Adds a layer to the message formation model

    set_aggregation(self, aggregation)
        Sets the aggregation strategy for the mp

    set_update_strategy(self, update_type, recurrent_type = 'GRU')
        Sets the update strategy for the mp by creating the appropriate model

    add_update_layer(self, **dict)
        Adds a layer to the update model
    """

    def __init__ (self, m):
        """
        Parameters
        ----------
        m:    dict
            Dictionary with the required attributes
        """

        self.type = m['type']
        self.destination_entity = m['destination_entity']
        self.source_entity = m['source_entity']
        self.adj_vector = m['adj_vector']
        self.extra_parameters = 0


        if 'aggregation' in m:
            self.aggregation = m['aggregation']

        if 'update' in m:
            self.update = self.create_update(m['update'])

        if 'message' in m:
            self.message_formation = self.create_message_formation(m['message'])

            #self.formation_type = self.find_type_of_message_creation(m['message_formation']['message_neural_net'])

            #if 'number_extra_parameters' in m['message_formation']:
            #    self.extra_parameters = m['message_formation']['number_extra_parameters']

            #if self.formation_type == 'feed_forward':
            #    self.message_formation = self.create_message_formation(m['message_formation'])


    def create_update(self, u):
        if u["type"] == 'apply_nn':
            return Apply_nn({'input': ['destination', 'aggregated'], 'architecture': u['architecture']})

        if u['type'] == 'apply_rnn':
            return Apply_rnn(u)


    # def create_update(self,dict):
    #     """
    #     Parameters
    #     ----------
    #     dict:    dict
    #         Dict with the parameters to be passed to the specific update model
    #     """
    #
    #     if dict['update_type'] == 'recurrent':
    #         parameters = dict.copy()
    #         del parameters['update_type']
    #         del parameters['recurrent_type']
    #
    #         parameters['name'] = self.destination_entity + '_update'
    #
    #         return Recurrent_Cell(dict['recurrent_type'], parameters)
    #
    #     elif dict['update_type'] == 'feed_forward':
    #         f = Feed_forward_model({'architecture':dict['architecture']}, model_role="update")
    #         return f


    def find_type_of_message_creation(self, type):
        """
        Parameters
        ----------
        type:    str
            Indicates if it uses a feed-forward, or the message is simply its hidden state
        """

        if type == 'yes':
            return 'feed_forward'
        return 'hidden_state'


    def create_message_formation(self, operations):
        result = []
        for op in operations:
            if op['type'] == 'apply_nn':
                print("adjslfjdsalfj",op)
                result.append(Apply_nn(op))

            if op['type'] == 'None':
                result.append(Operation("None"))
        return result


    # def create_message_formation(self, dict):
    #     """
    #     Parameters
    #     ----------
    #     dict:    dict
    #         Dict with the parameters to pass to the feed-forward architecture
    #     """
    #
    #     if 'number_extra_parameters' in dict:
    #         self.extra_parameters = dict['number_extra_parameters']
    #
    #
    #     f = Feed_forward_message_creation(dict['architecture'], self.extra_parameters)
    #     return f


    def get_instance_info(self):
        ordered = str((self.aggregation == 'ordered') or (self.aggregation == 'combination') or (self.aggregation == 'attention'))

        return [self.adj_vector, self.source_entity, self.destination_entity, ordered, str(self.extra_parameters > 0)]

    #SETTERS
    def set_message_formation(self, message_neural_net, number_extra_parameters = 0):
        """
        Parameters
        ----------
        message_neural_net:    str
            If it uses a feed_forward model or not
        number_extra_parameters:    int
            Extra parameters to be used from the dataset
        """

        if  message_neural_net == 'no':
            self.formation_type = 'recurrent'
        else:
            self.formation_type = 'feed_forward'
            self.message_formation= Feed_forward_message_creation({}, number_extra_parameters)


    def add_message_formation_layer(self, **dict):
        """
        Parameters
        ----------
        dict:    dict
            Dict with the information to create a given layer
        """

        self.message_formation.add_layer_aux(dict)

    def set_aggregation(self, aggregation):
        """
        Parameters
        ----------
        aggregation:    str
            Aggreagation type of the message passing
        """

        self.aggregation = aggregation


    def set_update_strategy(self, update_type, recurrent_type = 'GRU'):
        """
        Parameters
        ----------
        update_type:    str
            Wheteher its uses recurrent or feed-forward model for the update
        recurrent_type:    str (optional)
            What kind of recurrent model must be used.
        """

        self.update_type = update_type
        parameters = {}
        if update_type == 'recurrent':
            parameters['name'] = self.destination_entity + '_update'
            self.update = Recurrent_Cell(recurrent_type, parameters)

        elif update_type == 'feed_forward':
            self.update = Feed_forward_model({'architecture':{}},model_role="update")


    def add_update_layer(self, **dict):
        self.update.add_layer_aux(dict)


class Recurrent_Cell:
    """
    Attributes
    ----------
    type:    str
        Type of recurrent model to be used
    params:     dict
        Additional parameters


    Methods:
    --------
    get_tensorflow_object(self, destination_dimension
        Returns a tensorflow object with of this recurrent type with the destination_dimension as the number of units
    """

    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            Type of recurrent model to be used
        parameters:    dict
           Additional parameters of the model
        """

        self.type = type
        self.parameters = parameters


    def get_tensorflow_object(self, destination_dimension):
        """
        Parameters
        ----------
        destination_dimension:    int
            Number of units that the recurrent cell will have
        """
        self.parameters['units'] = destination_dimension
        c_ = getattr(tf.keras.layers, self.type+'Cell')
        layer = c_(**self.parameters)
        return layer


class Feed_forward_Layer:
    """
    Attributes
    ----------
    type:    str
        Type of recurrent model to be used
    params:     dict
        Additional parameters


    Methods:
    --------
    get_tensorflow_object(self, l_previous)
        Returns a tensorflow object of the containing layer, and sets its previous layer.

    get_tensorflow_object_last(self, l_previous, destination_units)
        Returns a tensorflow object of the last layer of the model, and sets its previous layer and the number of output units for it to have.

    """

    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            ?
        parameters:    dict
            Additional parameters of the model
        """

        self.type = type
        self.parameters = parameters
        if 'kernel_regularizer' in parameters:
            parameters['kernel_regularizer'] = tf.keras.regularizers.l2(float(parameters['kernel_regularizer']))

        if 'activation' in parameters and parameters['activation'] == 'None':
            parameters['activation'] = None

    def get_tensorflow_object(self, l_previous):
        """
        Parameters
        ----------
        l_previous:    object
            Previous layer of the architecture
        """

        c_ = getattr(tf.keras.layers, self.type)
        layer = c_(**self.parameters)(l_previous)
        return layer

    def get_tensorflow_object_last(self, l_previous, destination_units):
        """
        Parameters
        ----------
        l_previous:    object
            Previous layer of the architecture
        destination_dimension:    int
            Number of units that the recurrent cell will have
        """

        c_ = getattr(tf.keras.layers, self.type)
        self.parameters['units'] = destination_units
        layer = c_(**self.parameters)(l_previous)
        return layer


class Feed_forward_model:
    """
    Attributes:
    ----------
    layers:    array
        Layers contained in this feed-forward
    counter:    int
        Counts the current number of layers


    Methods:
    --------
    add_layer(self, **l)
        Adds a layer to the model

    add_layer_aux(self,l)
        Adds a layer to the model

    """

    def __init__(self, model,model_role):
        """
        Parameters
        ----------
        model:    dict
            Information regarding the architecture of the feed-forward
        """

        self.layers = []
        self.counter = 0

        if 'architecture' in model:
            dict = model['architecture']
            for l in dict:
                type_layer = l['type_layer'] #type of layer
                if 'name' not in l:
                    l['name'] = 'layer_' + str(self.counter) + '_' + type_layer + '_' + str(model_role)
                del l['type_layer']  #leave only the parameters of the layer

                layer = Feed_forward_Layer(type_layer, l)
                self.layers.append(layer)
                self.counter +=1

    def add_layer(self, **l):
        """
        Parameters
        ----------
        l:    dict
            Information of the new layer to be added
        """

        type_layer = l['type_layer']
        del l['type_layer']
        layer = Feed_forward_Layer(type_layer, l)
        self.layers.append(layer)

    def add_layer_aux(self,l):
        """
        Parameters
        ----------
        l:    dict
            Information of the new layer to be added
        """

        type_layer = l['type_layer']
        del l['type_layer']
        layer = Feed_forward_Layer(type_layer, l)
        self.layers.append(layer)


class Feed_forward_message_creation(Feed_forward_model):
    """
    Attributes:
    ----------
    architecture:    dict
        Architecture information of the model
    num_parameter:    int
        Extra parameters to be used from the dataset
    """

    def __init__(self,architecture, num_parameter):
        """
        Parameters
        ----------
        architecture:    dict
            Architecture information of the model
        num_parameter:    int
            Extra parameters to be used from the dataset
        """

        super(Feed_forward_message_creation, self).__init__({'architecture':architecture}, model_role="message_creation")
        self.num_extra_parameters = num_parameter


class Readout_model(Feed_forward_model):
    """
    Attributes
    ----------
    type:   str
        Type of message passing (individual or combined)
    entity:     str
        Name of the destination entity which shall be used for the predictions
    output_label:   str
        Name found in the dataset with the labels to be predicted
    output_normalization:   str (opt)
        Normalization to be used for the labels and predictions
    output_denormalization:     str (opt)
        Denormalization strategy for the labels and predictions
    """

    def __init__(self, output):
        """
        Parameters
        ----------
        output:    dict
            Dictionary with the readout_model parameters
        """

        super(Readout_model, self).__init__(output, model_role="readout")
        self.type = output['type_of_prediction']
        self.entity = output['entity']
        self.output_label = output['output_label']
        self.output_normalization = None
        self.output_denormalization = None

        if 'output_normalization' in output:
            self.output_normalization = output['output_normalization']

        if 'output_denormalization' in output:
            self.output_denormalization = output['output_denormalization']