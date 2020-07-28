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
from generator_std_to_framework import generator
from main import *
from framework_operations import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
from keras import backend as K

def set_model_info(model_description):
    """
    Parameters
    ----------
    model_description:    object
        Object with the json information model
    """

    global model_info
    model_info = model_description


def normalization(x, feature_list, output_names, output_normalizations,y=None):
    """
    Parameters
    ----------
    x:    tensor
        Tensor with the feature information
    y:    tensor
        Tensor with the label information
    feature_list:    tensor
        List of names with the names of the features in x
    output_names:    tensor
        List of names with the name of the output labels in y
    output_normalizations: dict
        Maps each feature or label with its normalization strategy if any
    """

    for f in feature_list:
        f_name = f.name
        norm_type = f.normalization
        if str(norm_type) != 'None':
            try:
                x[f_name] = eval(norm_type)(x[f_name], f_name)
            except:
                tf.compat.v1.logging.error('IGNNITION: The normalization function ' + str(norm_type) + ' is not defined in the main file.')
                sys.exit(1)


    #this normalization is only ready for one single output !!!
    if y != None:   #if we have labels to normalize
        n = len(output_names)

        for i in range(n):
            norm_type = output_normalizations[i]

            if str(norm_type) != 'None':
                try:
                    y = eval(norm_type)(y, output_names[i])
                except:
                    tf.compat.v1.logging.error('IGNNITION: The normalization function ' + str(norm_type) + ' is not defined in the main file.')
                    sys.exit(1)
        return x, y


    return x

def input_fn(data_dir, shuffle=False, training = True):
    """
    Parameters
    ----------
    x:    tensor
        Tensor with the feature information
    y:    tensor
        Tensor with the label information
    feature_list:    tensor
        List of names with the names of the features in x
    output_names:    tensor
        List of names with the name of the output labels in y
    output_normalizations: dict
        Maps each feature or label with its normalization strategy if any
    """

    with tf.name_scope('get_data') as _:
        feature_list = model_info.get_all_features()
        adjecency_info = model_info.get_adjecency_info()
        entity_list = model_info.get_entities()
        interleave_list = model_info.get_interleave_tensors()
        interleave_sources = model_info.get_interleave_sources()
        output_names, output_normalizations,_ = model_info.get_output_info()
        additional_input = model_info.obtain_additional_input_names()
        unique_additional_input = [a for a in additional_input if a not in feature_list]

        types = {}
        shapes = {}
        feature_names = []

        for a in unique_additional_input:
            types[a] = tf.int64
            shapes[a] = tf.TensorShape(None)


        for f in feature_list:
            f_name = f.name
            feature_names.append(f_name)
            types[f_name] =tf.float32
            shapes[f_name] = tf.TensorShape(None)

        for a in adjecency_info:
            types['src_' + a[0]] = tf.int64
            shapes['src_' + a[0]] = tf.TensorShape([None])
            types['dst_' + a[0]] = tf.int64
            shapes['dst_' + a[0]] = tf.TensorShape([None])

            #indicates that we need to keep track of the order
            if a[3] == 'True':
                types['seq_' + a[1] + '_' + a[2]] = tf.int64
                shapes['seq_' + a[1] + '_' + a[2]] = tf.TensorShape([None])

            if a[4] == 'True':
                types['params_' + a[0] ] = tf.int64
                shapes['params_' + a[0]] = tf.TensorShape(None)

        for e in entity_list:
            types['num_'+e.name] = tf.int64
            shapes['num_' + e.name] = tf.TensorShape([])


        for i in interleave_sources:
            types['indices_' + i[0] + '_to_' + i[1]] = tf.int64
            shapes['indices_' + i[0] + '_to_' + i[1]] = tf.TensorShape([None])


        if training: #if we do training, we also expect the labels
            ds = tf.data.Dataset.from_generator(generator,
                                                (types, tf.float32),
                                                (shapes, tf.TensorShape(None)),
                                                args=(data_dir, feature_names, output_names, adjecency_info,interleave_list, unique_additional_input, training, shuffle))

        else:
            ds = tf.data.Dataset.from_generator(generator,
                                                (types),
                                                (shapes),
                                                args=(data_dir, feature_names, output_names, adjecency_info, interleave_list, unique_additional_input, training, shuffle))

        #ds = ds.batch(2)

        with tf.name_scope('normalization') as _:
            if not training:
                ds = ds.map(lambda x: normalization(x, feature_list, output_names, output_normalizations))
            else:
                ds = ds.map(lambda x, y: normalization(x, feature_list, output_names, output_normalizations, y))


        ds = ds.repeat()
        with tf.name_scope('create_iterator') as _:
            if training:
                ds = ds.prefetch(100)
            else:
                ds = tf.compat.v1.data.make_initializable_iterator(ds)

    return ds


def r_squared(labels, predictions):
    """
    Parameters
    ----------
    labels:    tensor
        Label information
    predictions:    tensor
        Predictions of the model
    """
    labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(predictions,[-1])
    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

    m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

    return m_r_sq, update_rsq_op


class ComnetModel(tf.keras.Model):
    def __init__(self):

        super(ComnetModel, self).__init__()
        self.input_dimensions = model_info.get_input_dimensions()
        self.instances_per_step = model_info.get_mp_instances()

        with tf.name_scope('model_initializations') as _:
            for step in self.instances_per_step:
                for message in step[1]:
                    #Creation of the message creation models
                    with tf.name_scope('message_creation_models') as _:
                        operations = message.message_formation

                        counter = 0
                        for operation in operations:
                            if operation.type == 'feed_forward_nn':
                                src_entity = message.source_entity
                                dst_entity = message.destination_entity
                                var_name = src_entity + "_to_" + dst_entity + '_message_creation_' + str(counter)
                                #find out what the input dimension is (need to keep track of previous ones)

                                #input_dimension = int(self.input_dimensions[src_entity] + self.input_dimensions[dst_entity] + message.message_formation.num_extra_parameters)

                                #Find out the dimension of the model
                                input_nn = operation.input
                                input_dimension = 0
                                for i in input_nn:
                                    if i == 'hs_source':
                                        input_dimension += int(self.input_dimensions[src_entity])
                                    elif i == 'hs_dest':
                                        input_dimension += int(self.input_dimensions[dst_entity])
                                    elif i == 'edge_params':
                                        input_dimension += int(message.extra_parameters)
                                    else:
                                        dimension = getattr(self, i + '_dim')
                                        input_dimension += dimension


                                setattr(self, str(var_name) + "_layer_" + str(0), tf.keras.Input(shape=(input_dimension,)))

                                layer_counter = 1
                                layers = operation.model.layers
                                output_shape = 0
                                for l in layers:
                                    l_previous = getattr(self, str(var_name) + "_layer_" + str(layer_counter - 1))

                                    try:
                                        layer_model = l.get_tensorflow_object(l_previous)
                                        setattr(self, str(var_name) + "_layer_" + str(layer_counter), layer_model)
                                    except:
                                        tf.compat.v1.logging.error('IGNNITION: The layer ' + str(layer_counter) + ' of the message creation neural network in the message passing from ' + message.source_entity + ' to ' + message.destination_entity +
                                                                ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                        sys.exit(1)

                                    output_shape = int(layer_model.shape[1])
                                    layer_counter += 1


                                #Need to keep track of the output dimension of this one, in case we need it for a new model
                                if operation.output_name != 'None':
                                    setattr(self, operation.output_name + '_dim', output_shape)


                                # Create the actual model. Seems like the final model doesn't recognize this automatically
                                setattr(self, var_name, tf.keras.Model(inputs=getattr(self, str(var_name) + "_layer_" + str(0)),
                                                    outputs=getattr(self,str(var_name) + "_layer_" + str(layer_counter - 1)),
                                                    name=var_name))


                            counter += 1

                            #DEFINE HERE MORE OPERATIONS FOR MESSAGE CRETION FUNCTION


                    #Creation of the update models
                    with tf.name_scope('update_models') as _:

                        #Treat the individual message passings
                        if message.type == 'single_source':
                            update_model = message.update
                            #create the recurrent update models
                            if update_model.type == "recurrent_nn":
                                dst_entity = message.destination_entity
                                recurrent_cell = update_model.model
                                try:
                                    recurrent_instance = recurrent_cell.get_tensorflow_object(self.input_dimensions[dst_entity])
                                    setattr(self, str(dst_entity)+'_update', recurrent_instance)
                                except:
                                    tf.compat.v1.logging.error(
                                        'IGNNITION: The definition of the recurrent cell in message passsing from ' + message.source_entity + ' to ' + message.destination_entity +
                                        ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                    sys.exit(1)


                            #create the feed-forward upddate models
                            else:
                                model = update_model.model
                                src_entity = message.source_entity
                                dst_entity = message.destination_entity
                                var_name = dst_entity + "_ff_update"

                                with tf.name_scope(dst_entity + '_ff_update') as _:

                                    #input is the aggregated hs of the sources concat with the current dest. hs
                                    input_dimension = int(self.input_dimensions[src_entity]) + int(self.input_dimensions[dst_entity])

                                    setattr(self, str(var_name) + "_layer_" + str(0), tf.keras.Input(shape=(input_dimension,)))

                                    layer_counter = 1
                                    layers = model.layers
                                    n_layers = len(layers)
                                    for j in range(n_layers):
                                        l = layers[j]
                                        l_previous = getattr(self, str(var_name) + "_layer_" + str(layer_counter - 1))
                                        try:
                                            #if it's the last layer, set the output units to 1
                                            if j == n_layers -1:
                                                layer = l.get_tensorflow_object_last(l_previous, int(self.input_dimensions[dst_entity]))
                                            else:
                                                layer = l.get_tensorflow_object(l_previous)

                                            setattr(self, str(var_name) + "_layer_" + str(layer_counter), layer)
                                        except:
                                            tf.compat.v1.logging.error(
                                                'IGNNITION: The layer ' + str(layer_counter) + ' of the update neural network in message passing from ' + message.source_entity + ' to ' + message.destination_entity +
                                                ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                            sys.exit(1)

                                        layer_counter += 1

                                    #Create the final ff-model
                                    setattr(self, var_name,
                                            tf.keras.Model(inputs=getattr(self, str(var_name) + "_layer_" + str(0)),
                                                           outputs=getattr(self,str(var_name) + "_layer_" + str(layer_counter - 1)),
                                                           name=var_name ))



            #Create the update models for the combined mp
            combined_models = model_info.get_combined_mp_options()
            for step in combined_models:
                messages_to_combine = combined_models[step]
                for mp in messages_to_combine:
                    dst_entity = mp.destination_entity
                    model_op = mp.update

                    if model_op.type == 'recurrent_nn':
                        try:
                            #obtain the recurrent_models
                            recurrent_cell = model_op.model
                            recurrent_instance = recurrent_cell.get_tensorflow_object(self.input_dimensions[dst_entity])
                            setattr(self, str(dst_entity) + '_combined_update', recurrent_instance)

                        except:
                            tf.compat.v1.logging.error(
                                'IGNNITION: The definition of the recurrent cell in the combined message passsing with destination entity ' + '"' + dst_entity + '"' )
                            sys.exit(1)

                    else:   #if it's a feed_forward
                        try:
                            dst_entity = mp.destination_entity

                            #need to calculate the input size (size of the messages)    #DO THIS WHEN CREATING THE CLASS?
                            sources = model_info.get_combined_mp_sources(dst_entity, step)
                            if mp.message_combination == 'concat' and mp.concat_axis ==2:   #if we are concatenating by message
                                message_dimensionality = 0
                                for s in sources:
                                    message_dimensionality += int(self.input_dimensions[s])


                            else:   #all the messages are the same size. So take the first for instance
                                message_dimensionality = int(self.input_dimensions[sources[0]])


                            var_name = dst_entity + "_ff_combined_update"

                            with tf.name_scope(dst_entity + '_ff_update') as _:

                                # input is the aggregated hs of the sources concat with the current dest. hs
                                input_dimension = message_dimensionality + int(
                                    self.input_dimensions[dst_entity])

                                setattr(self, str(var_name) + "_layer_" + str(0),
                                        tf.keras.Input(shape=(input_dimension,)))
                                layer_counter = 1
                                layers = model_op.model.layers
                                n_layers = len(layers)
                                for j in range(n_layers):
                                    l = layers[j]
                                    l_previous = getattr(self, str(var_name) + "_layer_" + str(layer_counter - 1))
                                    try:
                                        # if it's the last layer, set the output units to 1
                                        if j == n_layers - 1:
                                            layer = l.get_tensorflow_object_last(l_previous, int(
                                                self.input_dimensions[dst_entity]))
                                        else:
                                            layer = l.get_tensorflow_object(l_previous)

                                        setattr(self, str(var_name) + "_layer_" + str(layer_counter), layer)
                                    except:
                                        tf.compat.v1.logging.error(
                                            'IGNNITION: The layer ' + str(
                                                layer_counter) + ' of the update neural network in message passing from ' + mp.source_entity + ' to ' + mp.destination_entity +
                                            ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                        sys.exit(1)

                                    layer_counter += 1

                                # Create the final ff-model
                                setattr(self, var_name,
                                        tf.keras.Model(inputs=getattr(self, str(var_name) + "_layer_" + str(0)),
                                                       outputs=getattr(self, str(var_name) + "_layer_" + str(
                                                           layer_counter - 1)),
                                                       name=var_name))


                        except:
                            tf.compat.v1.logging.error(
                                'IGNNITION: The definition of the neural network in the combined message passsing with destination entity ' + '"' + dst_entity + '"')
                            sys.exit(1)


            #Create the several readout models
            readout_operations = model_info.get_readout_operations()
            counter = 0
            for operation in readout_operations:
                if operation.type == "predict" or operation.type == 'neural_network':
                    with tf.name_scope("readout_architecture"):

                        #input_dimension = int(self.input_dimensions[operation.input[0]])
                        input_dimension = 0
                        for i in operation.input:
                            input_dimension += int(self.input_dimensions[i])

                        setattr(self, 'readout_model' + str(counter)+"_layer_"+str(0), tf.keras.Input(shape = (input_dimension,)))

                        layer_counter = 1
                        layers = operation.architecture.layers
                        for l in layers:
                            l_previous = getattr(self, 'readout_model' + str(counter)+"_layer_"+str(layer_counter - 1))
                            try:
                                layer = l.get_tensorflow_object(l_previous)
                                setattr(self, 'readout_model' + str(counter)+"_layer_"+str(layer_counter), layer)
                            except:
                                tf.compat.v1.logging.error(
                                    'IGNNITION: The layer ' + str(layer_counter) + ' of the readout is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                sys.exit(1)

                            layer_counter += 1

                        #Create the actual model with all the previous layers.
                        model = tf.keras.Model(inputs=getattr(self,'readout_model' + str(counter)+"_layer_"+str(0)), outputs=getattr(self, 'readout_model' + str(counter)+"_layer_"+str(layer_counter -1)))
                        setattr(self, 'readout_model_' +str(counter),model)


                    #save the dimensions of the output
                    if operation.type == 'neural_network':
                        self.input_dimensions[operation.output_name] = model.layers[-1].output.shape[1]


                elif operation.type == 'pooling':
                    if operation.type_pooling == 'sum' or operation.type_pooling=='max' or operation.type_pooling=='mean':
                        dimensionality = self.input_dimensions[operation.input[0]]

                    else:
                        dimensionality = 1

                    self.input_dimensions[operation.output_name] = dimensionality   #add the new dimensionality to the input_dimensions tensor


                elif operation.type == 'product':
                    self.input_dimensions[operation.output_name] = self.input_dimensions[operation.input[0]]

                elif operation.type == 'extend_adjacencies':
                    self.input_dimensions[operation.output_name[0]] = self.input_dimensions[operation.input[0]]
                    self.input_dimensions[operation.output_name[1]] = self.input_dimensions[operation.input[1]]



                counter += 1


    def call(self, input, training=False):
        """
        Parameters
        ----------
        input:    dict
            Dictionary with all the tensors with the input information of the model
        """

        # -----------------------------------------------------------------------------------
        # HIDDEN STATE CREATION
        entities = model_info.entities

        #Initialize all the hidden states for all the nodes.
        with tf.name_scope('hidden_states') as _:
            for entity in entities:
                name = entity.name

                with tf.name_scope('hidden_state_'+str(name)) as _:
                    dim = entity.hidden_state_dimension

                    first = True
                    features = entity.features
                    total = 0
                    #concatenate all the features
                    for feature in features:
                        name_feature = feature.name
                        total += feature.size

                        with tf.name_scope('add_feature_' + str(name_feature)) as _:
                            size = feature.size
                            input[name_feature] = tf.reshape(input[name_feature], tf.stack([input['num_'+str(name)], size]))

                            if first:
                                state = input[name_feature]
                                first = False
                            else:
                                state = tf.concat([state,input[name_feature]], axis=1, name="add_"+name_feature)


                        shape = tf.stack([input['num_' + name], dim - total], axis=0)  # shape (2,)

                        #add 0s until reaching the given dimension
                        with tf.name_scope('add_zeros_to_' + str(name)) as _:
                            state = tf.concat([state,tf.zeros(shape)], axis=1, name="add_zeros_"+name)
                            setattr(self, str(name)+"_state", state)



        # -----------------------------------------------------------------------------------
        # MESSAGE PASSING PHASE
        with tf.name_scope('message_passing') as _:

            for j in range(model_info.get_mp_iterations()):

                 with tf.name_scope('iteration_'+str(j)) as _:

                     for step in self.instances_per_step:
                         step_name = step[0]

                         #given one message from a given step
                         for message in step[1]:
                             src_name = message.source_entity
                             dst_name = message.destination_entity

                             with tf.name_scope(src_name + 's_to_' + dst_name + 's') as _:

                                 #create the messages
                                 sources = input['src_' + message.adj_vector]
                                 destinations = input['dst_' + message.adj_vector]
                                 states = getattr(self, str(src_name) + '_state')
                                 num_dst = input['num_' + dst_name]
                                 messages = tf.gather(states,sources)  # initially all the messages are the source hs itself
                                 src_hs = tf.gather(states,sources)   # obtain each of the source hs for each adj of the mp
                                 dst_hs = getattr(self, dst_name + '_state')

                                 #use a ff if so needed
                                 with tf.name_scope('create_message_' + src_name + '_to_' + dst_name) as _:
                                     message_creation_models = message.message_formation

                                     counter = 0
                                     for model in message_creation_models:

                                        if model.type == 'feed_forward_nn':
                                            with tf.name_scope('apply_nn_' + str(counter)) as _:
                                                var_name = src_name + "_to_" + dst_name + '_message_creation_' + str(counter)  #careful. This name could overlap with another model
                                                message_creator = getattr(self, var_name)
                                                first = True
                                                with tf.name_scope('create_input') as _:
                                                    for i in model.input:
                                                        if i == 'hs_source':
                                                            if first:
                                                                input_nn = src_hs
                                                                first = False

                                                            else:
                                                                input_nn = tf.concat([input_nn, src_hs], axis = 1)

                                                        elif i == 'hs_dest':
                                                            h_states_dest = tf.gather(dst_hs, destinations)

                                                            if first:
                                                                input_nn = h_states_dest
                                                                first = False

                                                            else:
                                                                input_nn = tf.concat([input_nn, h_states_dest], axis=1)


                                                        elif i == 'edge_params':
                                                            edge_params = tf.cast(input['params_' + message.adj_vector], tf.float32)

                                                            if first:
                                                                input_nn = edge_params
                                                                first = False
                                                            else:
                                                                input_nn = tf.concat([input_nn, edge_params], axis=1)

                                                        else:   #it is the output of a previous operation
                                                            other_tensor = getattr(self, i + '_var')
                                                            if first:
                                                                input_nn = other_tensor
                                                                first = False
                                                            else:
                                                                input_nn = tf.concat([input_nn, other_tensor], axis=1)


                                                with tf.name_scope('create_message') as _:
                                                    result = message_creator(input_nn)
                                                    if model.output_name != 'None':
                                                        setattr(self, model.output_name + '_var', result)

                                                    messages = result   #by default, the message is always the result from the last operation


                                        counter +=1


                                 #treating the individual message passings
                                 if message.type == "single_source":

                                     #-----------------------------------------
                                     #aggregation

                                     aggregation = message.aggregation

                                     #sum aggregation
                                     if aggregation == 'sum':
                                         with tf.name_scope("aggregate_sum_" + src_name) as _:
                                             source_input = tf.math.unsorted_segment_sum(messages, destinations, num_dst)

                                     #ordered aggregation
                                     elif aggregation == 'ordered':
                                         with tf.name_scope("aggregate_ordered_" + src_name) as _:
                                             seq = input['seq_' + src_name + '_' + dst_name]

                                             ids = tf.stack([destinations, seq],axis=1)  # stack of pairs of the path value and its sequence value

                                             max_len = tf.reduce_max(seq) + 1

                                             shape = tf.stack([num_dst, max_len, int(self.input_dimensions[src_name])])  # shape(n_paths, max_len_path, dimension_link)

                                             lens = tf.math.unsorted_segment_sum(tf.ones_like(destinations),
                                                                                 destinations,
                                                                                 num_dst)

                                             source_input = tf.scatter_nd(ids, messages, shape)

                                     #attention aggreagation
                                     elif aggregation == 'attention':
                                         with tf.name_scope("attention_mechanism_" + src_name) as _:
                                             #obtain the source states  (NxF1)
                                             h_src = tf.identity(messages)
                                             F1 = int(self.input_dimensions[message.source_entity])

                                             #obtain the destination states  (NxF2)
                                             states_dest = getattr(self, str(dst_name) + '_state')
                                             h_dst = tf.gather(states_dest, destinations)
                                             F2 = int(self.input_dimensions[message.destination_entity])

                                             #new number of features
                                             F_ = F1

                                             #now apply a linear transformation for the sources (NxF1 -> NxF')
                                             kernel1 = self.add_weight(shape=(F1, F_))
                                             transformed_states_sources = K.dot(h_src, kernel1)  # NxF'   (W h_i for every source)

                                             # now apply a linear transformation for the destinations (NxF2 -> NxF')
                                             kernel2 = self.add_weight(shape=(F2, F_))
                                             transformed_states_dest = K.dot(h_dst, kernel2)  # NxF'   (W h_i for every dst)

                                             # concat source and dest for each edge
                                             attention_input = tf.concat([transformed_states_sources, transformed_states_dest], axis = 1)   #Nx2F'

                                             #apply the attention weight vector    (N x 2F_) * (2F_ x 1) = (N x 1)
                                             attn_kernel = self.add_weight(shape=(2*F_, 1))
                                             attention_input = K.dot(attention_input, attn_kernel)  #Nx1


                                             #apply the non linearity
                                             attention_input = tf.keras.layers.LeakyReLU(alpha=0.2)(attention_input)

                                             #reshape into a matrix where every row is a destination node and every column is one of its neighbours
                                             seq = input['seq_' + src_name + '_' + dst_name]
                                             ids = tf.stack([destinations, seq],axis=1)
                                             max_len = tf.reduce_max(seq) + 1
                                             shape = tf.stack([num_dst, max_len, 1])
                                             aux = tf.scatter_nd(ids, attention_input, shape)


                                             #apply softmax to it (by rows)
                                             coeffients = tf.keras.activations.softmax(aux, axis = 0)

                                             #sum them all together using the coefficients (average)
                                             final_coeffitients = tf.gather_nd(coeffients, ids)
                                             weighted_inputs = messages * final_coeffitients
                                             source_input = tf.math.unsorted_segment_sum(weighted_inputs, destinations,num_dst)

                                     #convolutional aggregation
                                     elif message.aggregation == 'convolutional':
                                         print("Here we would do a convolution")


                                     #---------------------------------------
                                     #update

                                     update_model = message.update

                                     #recurrent update
                                     if update_model.type == "recurrent_nn":
                                         aggregation = message.aggregation

                                         #if the aggregation was a sum
                                         if aggregation == 'sum' or aggregation == 'attention' or aggregation=='convolutional':
                                             with tf.name_scope("update_sum_" + dst_name) as _:
                                                 old_state = getattr(self, str(dst_name)+'_state')
                                                 model = getattr(self,str(dst_name)+'_update')

                                                 new_state, _ = model(source_input, [old_state])
                                                 setattr(self, str(dst_name)+'_state', new_state)


                                         #if the aggregation was ordered
                                         elif aggregation == 'ordered':
                                             with tf.name_scope("update_ordered_" + dst_name) as _:
                                                 old_state = getattr(self, str(dst_name) + '_state')
                                                 model = getattr(self, str(dst_name) + '_update')
                                                 gru_rnn = tf.keras.layers.RNN(model, return_sequences=True,
                                                                               return_state=True, name = str(dst_name)+'_update')
                                                 outputs, new_state = gru_rnn(inputs=source_input, initial_state=old_state,mask=tf.sequence_mask(lens))
                                                 setattr(self, str(dst_name) + '_state', new_state)


                                     #feed-forward update:
                                     #restriction: It can only be used if the aggreagation was ordered.
                                     elif update_model.type == 'feed_forward_nn':
                                         update = getattr(self, dst_name + '_ff_combined_update')
                                         current_hs = getattr(self, dst_name + '_state')

                                         # now we need to obtain for each adjacency the concatenation of the source and the destination
                                         update_input = tf.concat([source_input, current_hs], axis = 1)
                                         new_state = update(update_input)
                                         setattr(self, str(dst_name) + '_state', new_state) #update new state



                                #Treat the combined message passings' aggregation. Only pre-processing
                                 else:

                                     #--------------------------------------------
                                     #aggregation

                                     #sum aggreagtion
                                     #if message.aggregation == 'sum':  # <---- This is not callable yet
                                     #    with tf.name_scope('combined_sum_preprocessing' + src_name) as _:
                                     #        m = tf.math.unsorted_segment_sum(messages, destinations,num_dst)  # m is the aggregated values. Sum together the values belonging to the same path
                                     #        #setattr(self, str(src_name) + '_sum_combined', m)
                                     #        setattr(self, str(src_name) + '_to_' + str(dst_name) + '_combined', m)
                                     #        setattr(self, 'lens_' + str(src_name), lens)


                                     #combination aggreagation
                                     if message.aggregation == 'combination':
                                         with tf.name_scope('combination_preprocessing' + src_name) as _:
                                             seq = input['seq_' + src_name + '_' + dst_name]

                                             ids = tf.stack([destinations, seq],axis=1)

                                             lens = tf.math.segment_sum(data=tf.ones_like(destinations),segment_ids=destinations)

                                             max_len = tf.reduce_max(seq) + 1

                                             shape = tf.stack([num_dst, max_len, int(self.input_dimensions[src_name])])

                                             source_input = tf.scatter_nd(ids, messages, shape) #find the input ordering it by sequence

                                             setattr(self, str(src_name) + '_to_' + str(dst_name) +  '_combined', source_input)

                                             setattr(self, 'lens_' + str(src_name), lens)



                         #---------------------------------
                         #Combined message passings

                         #update for each of the combined message passing
                         combined_models = model_info.get_combined_mp_options() #improve

                         if step_name in combined_models:
                             messages_to_combine = combined_models[step_name]
                             for message in messages_to_combine:
                                 dst_name = message.destination_entity
                                 #entities that are senders in this message passing
                                 combine_sources = model_info.get_combined_mp_sources(dst_name,step_name)

                                 #concatenation combination.
                                 with tf.name_scope("combined_mp_to" + dst_name) as _:

                                     if message.message_combination == 'concat':
                                         with tf.name_scope("concatenate_input_of_" + dst_name) as _:

                                             first = True
                                             for src_name in combine_sources:
                                                 len = getattr(self, 'lens_' + str(src_name))  # this is the vector with lens of each destination

                                                 with tf.name_scope("concat_" + src_name) as _:
                                                     #obtain the new states. Only one message from a certain entity type to each destination node
                                                     state = getattr(self, str(src_name) + '_to_' + str(dst_name) + '_combined')    #destinations x max_num_sources x dim_src

                                                     if first:
                                                         sources_input = state
                                                         first = False
                                                         final_len = len
                                                     else:
                                                         #concatenate, so every node receives only one message
                                                         #axis = 1 is to concat all the messages for a dest. axis=2 is to concat by message
                                                         sources_input = tf.concat([sources_input, state], axis=message.concat_axis)

                                                         if message.concat_axis == 1:   #if axis=2, then the number of messages received is the same. Simply create bigger messages
                                                             final_len += len

                                     #interleave combination
                                     elif message.message_combination == 'interleave':
                                         with tf.name_scope('interleave_' + dst_name) as _:

                                             first = True
                                             for src_name in combine_sources:
                                                 with tf.name_scope('add_' + src_name) as _:
                                                     len = getattr(self, 'lens_' + str(src_name))   #this is the vector with lens of each destination

                                                     s = getattr(self, str(src_name) + '_to_' + str(dst_name) +  '_combined')  #input

                                                     #form the pattern to be used for the interleave
                                                     indices_source = input["indices_" + src_name + '_to_' + dst_name]
                                                     if first:
                                                         first = False
                                                         sources_input = s  # destinations x max_of_sources_to_dest x dim_source
                                                         indices = indices_source
                                                         final_len = len
                                                     else:
                                                         sources_input = tf.concat([sources_input, s], axis = 1)   # destinations x max_of_sources_to_dest_concat x dim_source
                                                         indices = tf.stack([indices, indices_source], axis = 0)
                                                         final_len = tf.math.add(final_len, len)    #check this len? We just sum the max of each of them?

                                             #place each of the messages into the right spot in the sequence defined
                                             with tf.name_scope('order_sources_from_' + dst_name) as _:
                                                 sources_input = tf.transpose(sources_input, perm =[1,0,2]) # destinations x max_of_sources_to_dest_concat x dim_source ->  (max_of_sources_to_dest_concat x destinations x dim_source)
                                                 indices = tf.reshape(indices, [-1, 1])

                                                 sources_input = tf.scatter_nd(indices, sources_input, tf.shape(sources_input, out_type=tf.int64))

                                                 sources_input = tf.transpose(sources_input, perm=[1, 0, 2])    #(max_of_sources_to_dest_concat x destinations x dim_source) -> destinations x max_of_sources_to_dest_concat x dim_source


                                     #here we do the combined update
                                     update_model = message.update
                                     if update_model.type == "recurrent_nn":
                                         with tf.name_scope('recurrent_update_' + dst_name) as _:
                                             old_state = getattr(self, str(dst_name) + '_state')
                                             model = getattr(self, str(dst_name) + '_combined_update')

                                             gru_rnn = tf.keras.layers.RNN(model, return_sequences=True,return_state=True)
                                             outputs, new_state = gru_rnn(inputs=sources_input, initial_state=old_state,mask=tf.sequence_mask(final_len))

                                             setattr(self, str(dst_name) + '_state', new_state)


                                     else:  #if feed_forward. ASK
                                         with tf.name_scope('ff_update_' + dst_name) as _:
                                             try:
                                                 update = getattr(self, dst_name + '_ff_combined_update')
                                                 current_hs = getattr(self, dst_name + '_state')

                                                 # now we need to obtain for each adjacency the concatenation of the source and the destination
                                                 update_input = tf.concat([sources_input, current_hs], axis=1)
                                                 new_state = update(update_input)
                                                 setattr(self, str(dst_name) + '_state', new_state)  # update new state

                                             except:
                                                 tf.compat.v1.logging.error(
                                                     'IGNNITION:  This functionality is not yet fully supported')
                                                 sys.exit(1)


        # -----------------------------------------------------------------------------------
        #READOUT PHASE
        with tf.name_scope('readout_predictions') as _:
            readout_opeartions = model_info.get_readout_operations()

            counter = 0
            for operation in readout_opeartions:
                if operation.type == "predict":
                    model = getattr(self, 'readout_model_' +str(counter))

                    try:
                        # if we are reusing information
                        prediction_input = getattr(self, operation.input[0] + '_state')
                    except:
                        # if it is some additional information of the dataset
                        prediction_input = input[operation.input[0]]

                    r = model(prediction_input)    #predicting should only be done once.

                    if operation.output_name is not None:
                        setattr(self, operation.output_name + '_state', r)
                    return r


                if operation.type == "pooling":
                    #obtain the input of the pooling operation
                    try:
                        pooling_input = getattr(self, operation.input[0] + '_state')
                    except:
                        pooling_input = input[operation.input[0]]

                    #here we do the pooling
                    if operation.type_pooling == 'sum':

                        pooling_output = tf.reduce_sum(pooling_input, 0)
                        pooling_output = tf.reshape(pooling_output, [-1] + [pooling_output.shape.as_list()[0]])

                    elif operation.type_pooling == 'mean':
                        pooling_output = tf.reduce_mean(pooling_input,0)
                        pooling_output = tf.reshape(pooling_output, [-1] + [pooling_output.shape.as_list()[0]])

                    elif operation.type_pooling == 'max':
                        pooling_output = tf.reduce_max(pooling_input,0)
                        pooling_output = tf.reshape(pooling_output, [-1] + [pooling_output.shape.as_list()[0]])


                elif operation.type == 'product':
                    try:
                        product_input1 = getattr(self, operation.input[0] + '_state')
                    except:
                        product_input1 = input[operation.input[0]]


                    #obtain the second input
                    try:
                        product_input2 = getattr(self, operation.input[1] + '_state')
                    except:
                        product_input2 = input[operation.input[1]]

                    try:
                        if operation.type_product == 'dot_product':
                            product_output = tf.tensordot(product_input1, product_input2, axes = 0)

                        elif operation.type_product == 'element_wise':
                            product_output = tf.matmul(product_input1, product_input2)

                        setattr(self, operation.output_name + '_state', product_output)

                    except:
                        tf.compat.v1.logging.error('IGNNITION:  The product operation between ' + str(
                            operation.input[0]) + ' and ' + operation.input[1] + ' failed. Check that the dimensions are compatible.')
                        sys.exit(1)




                elif operation.type == 'neural_network':
                    var_name = 'readout_model_' +str(counter)
                    readout_nn = getattr(self, var_name)

                    first = True
                    for i in operation.input:
                        try:
                            # if we are reusing information
                            new_input = getattr(self, i + '_state')
                        except:
                            # if it is some additional information of the dataset (take it from the input information of the model)
                            new_input = input[i]

                        if first:
                            resulting_input = new_input
                            first = False
                        else:
                            resulting_input = tf.concat([resulting_input, new_input], axis=1)


                    result = readout_nn(resulting_input)
                    setattr(self, operation.output_name + '_state', result)


                elif operation.type == 'extend_adjacencies':
                    adj_list_src = input['src_' + operation.adj_list]
                    adj_list_dst = input['dst_' + operation.adj_list]

                    #get the source input
                    try:
                        source = getattr(self, operation.input[0] + '_state')
                    except:
                        source = input[operation.input[0]]

                    #get the destination input
                    try:
                        dest = getattr(self, operation.input[1] + '_state')
                    except:
                        dest = input[operation.input[1]]


                    #obtain the extended input (by extending it to the number of adjacencies between them)
                    try:
                        extended_src = tf.gather(source, adj_list_src)
                    except:
                        tf.compat.v1.logging.error('IGNNITION:  Extending the adjacency list ' + str(adj_list) + ' was not possible. Check that the indexes of the source of the adjacency list match the input given.')
                        sys.exit(1)

                    try:
                        extended_dst = tf.gather(dest, adj_list_dst)
                    except:
                        tf.compat.v1.logging.error('IGNNITION:  Extending the adjacency list ' + str(adj_list) + ' was not possible. Check that the indexes of the destination of the adjacency list match the input given.')
                        sys.exit(1)


                    #save the source
                    setattr(self, operation.output_name[0] + '_state', extended_src)

                    #save the destination
                    setattr(self, operation.output_name[1] + '_state', extended_dst)


                counter += 1




def model_fn(features,labels,mode):
    """
    Parameters
    ----------
    features:    dict
        All the features to be used as input
    labels:    tensor
        Tensor with the label information
    mode:    tensor
        Either train, eval or predict
    """

    #create the model
    model = ComnetModel()

    #peform the predictions
    predictions = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

    #prediction mode. Denormalization is done if so specified
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_names, _, output_denormalizations = model_info.get_output_info()  # for now suppose we only have one output type

        try:
            predictions = eval(output_denormalizations[0])(predictions, output_names[0])
        except:
            tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_names[0] + ' was not defined. The output will be normalized.')


        return tf.estimator.EstimatorSpec(
            mode, predictions= {
                'predictions': predictions
        })

    #dynamically define the loss function from the keras documentation
    name = model_info.get_loss()
    loss = globals()[name]
    loss_function = loss()

    regularization_loss = sum(model.losses)

    loss = loss_function(labels, predictions)

    total_loss = loss + regularization_loss
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    #evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        #perform denormalization if defined
        output_names, _, output_denormalizations = model_info.get_output_info()
        try:
            labels = eval(output_denormalizations[0])(labels, output_names[0])
            predictions = eval(output_denormalizations[0])(predictions, output_names[0])
        except:
            tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_names[0] + ' was not defined. The output (and statistics) will use the normalized values.')

        #metrics calculations
        label_mean = tf.keras.metrics.Mean()
        _ = label_mean.update_state(labels)
        prediction_mean = tf.keras.metrics.Mean()
        _ = prediction_mean.update_state(predictions)
        mae = tf.keras.metrics.MeanAbsoluteError()
        _ = mae.update_state(labels, predictions)
        mre = tf.keras.metrics.MeanRelativeError(normalizer=tf.abs(labels))
        _ = mre.update_state(labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': label_mean,
                'prediction/mean': prediction_mean,
                'mae': mae,
                'mre': mre,
                'r-squared': r_squared(labels, predictions)
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    grads = tf.gradients(total_loss, model.trainable_variables)
    summaries = [tf.summary.histogram(var.op.name, var) for var in model.trainable_variables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    #dynamically define the optimizer
    optimizer_params = model_info.get_optimizer()
    op_type = optimizer_params['type']
    del optimizer_params['type']

    #dynamically define the adaptative learning rate if needed
    schedule = model_info.get_schedule()
    if schedule != {}:
        type = schedule['type']
        del schedule['type']  # so that only the parameters remain
        s = globals()[type]
        optimizer_params['learning_rate'] = s(**schedule)  # create an instance of the schedule class indicated by the user. Accepts any schedule from keras documentation

    o = globals()[op_type]
    optimizer = o(**optimizer_params)  # create an instance of the optimizer class indicated by the user. Accepts any loss function from keras documentation


    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

    logging_hook = tf.estimator.LoggingTensorHook(
        {"Loss": loss,
         "Total loss": total_loss}
    , every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )

