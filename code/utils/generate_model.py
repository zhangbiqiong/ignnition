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

        if norm_type != 'None':
            try:
                x[f_name] = eval(norm_type)(x[f_name], f_name)
            except:
                tf.compat.v1.logging.error('IGNNITION: The normalization function ' + norm_type + ' is not defined in the main file.')
                sys.exit(1)


    #this normalization is only ready for one single output !!!
    if y != None:   #if we have labels to normalize
        n = len(output_names)

        for i in range(n):
            norm_type = output_normalizations[i]

            if norm_type != 'None':
                try:
                    y = eval(norm_type)(y, output_names[i])
                except:
                    tf.compat.v1.logging.error('IGNNITION: The normalization function ' + norm_type + ' is not defined in the main file.')
                    sys.exit(1)
        return x, y


    return x

def tfrecord_input_fn(data_dir, shuffle=False, training = True):
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
        combination_sources = model_info.get_combined_sources()
        output_names, output_normalizations,_ = model_info.get_output_info()

        types = {}
        shapes = {}
        feature_names = []
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


        for i in combination_sources:
            types['indices_' + i[0] + '_to_' + i[1]] = tf.int64
            shapes['indices_' + i[0] + '_to_' + i[1]] = tf.TensorShape([None])


        if training: #if we do training, we also expect the labels
            ds = tf.data.Dataset.from_generator(generator,
                                                (types, tf.float32),
                                                (shapes, tf.TensorShape(None)),
                                                args=(data_dir, feature_names, output_names, adjecency_info,interleave_list, training, shuffle))

        else:
            ds = tf.data.Dataset.from_generator(generator,
                                                (types),
                                                (shapes),
                                                args=(data_dir, feature_names, output_names, adjecency_info, interleave_list,training, shuffle))

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
                ds = tf.compat.v1.data.make_initializable_iterator(ds)
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
        self.entities_dimensions = model_info.get_entities_dimensions()
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

                                #input_dimension = int(self.entities_dimensions[src_entity] + self.entities_dimensions[dst_entity] + message.message_formation.num_extra_parameters)

                                #Find out the dimension of the model
                                input_nn = operation.input
                                input_dimension = 0
                                for i in input_nn:
                                    if i == 'hs_source':
                                        input_dimension += int(self.entities_dimensions[src_entity])
                                    elif i == 'hs_dest':
                                        input_dimension += int(self.entities_dimensions[dst_entity])
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
                                    recurrent_instance = recurrent_cell.get_tensorflow_object(self.entities_dimensions[dst_entity])
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
                                    input_dimension = int(self.entities_dimensions[src_entity]) + int(self.entities_dimensions[dst_entity])

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
                                                layer = l.get_tensorflow_object_last(l_previous, int(self.entities_dimensions[dst_entity]))
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
                            recurrent_instance = recurrent_cell.get_tensorflow_object(self.entities_dimensions[dst_entity])
                            setattr(self, str(dst_entity) + '_combined_update', recurrent_instance)

                        except:
                            tf.compat.v1.logging.error(
                                'IGNNITION: The definition of the recurrent cell in the combined message passsing with destination entity ' + '"' + dst_entity + '"' )
                            sys.exit(1)



            #Create the several readout models
            outputs = model_info.get_output_models()
            model_counter = 0
            for output in outputs:
                model_counter += 1
                with tf.name_scope('readout_architecture' + str(model_counter)) as _:
                    var_name = output.entity + '_' + str(model_counter)
                    input_dimension = int(self.entities_dimensions[output.entity])

                    setattr(self, str(var_name)+"_layer_"+str(0), tf.keras.Input(shape = (input_dimension,)))

                    layer_counter = 1
                    layers = output.layers
                    for l in layers:
                        l_previous = getattr(self, str(var_name)+"_layer_"+str(layer_counter -1))
                        try:
                            layer = l.get_tensorflow_object(l_previous)
                            setattr(self, str(var_name)+"_layer_"+str(layer_counter), layer)
                        except:
                            tf.compat.v1.logging.error(
                                'IGNNITION: The layer ' + str(layer_counter) + ' of the readout is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                            sys.exit(1)

                        layer_counter += 1

                    #Create the actual model with all the previous layers.
                    setattr(self, 'output_' + str(model_counter) +'_'+str(var_name),
                         tf.keras.Model(inputs=getattr(self,str(var_name)+"_layer_"+str(0)), outputs=getattr(self, str(var_name)+"_layer_"+str(layer_counter -1)), name = output.output_label + '_predictor' ))



    def call(self, input, training=False):
        """
        Parameters
        ----------
        input:    dict
            Dictionary with all the tensors with the input information of the model
        """

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


        #proceed with the message passing phase
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


                                                with tf.name_scope('update') as _:
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
                                         with tf.name_scope("agregate_sum_" + src_name) as _:
                                             source_input = tf.math.unsorted_segment_sum(messages, destinations, num_dst)

                                     #ordered aggregation
                                     elif aggregation == 'ordered':
                                         with tf.name_scope("agregate_ordered_" + src_name) as _:
                                             seq = input['seq_' + src_name + '_' + dst_name]

                                             ids = tf.stack([destinations, seq],axis=1)  # stack of pairs of the path value and its sequence value

                                             max_len = tf.reduce_max(seq) + 1

                                             shape = tf.stack([num_dst, max_len, int(self.entities_dimensions[src_name])])  # shape(n_paths, max_len_path, dimension_link)

                                             lens = tf.math.unsorted_segment_sum(tf.ones_like(destinations),
                                                                                 destinations,
                                                                                 num_dst)  # destinations should be in order. CHECK!!

                                             source_input = tf.scatter_nd(ids, messages, shape)

                                     #attention aggreagation
                                     elif aggregation == 'attention':
                                         with tf.name_scope("attention_mechanism_" + src_name) as _:
                                             #obtain the source states  (NxF1)
                                             h_src = tf.identity(messages)
                                             F1 = int(self.entities_dimensions[message.source_entity])

                                             #obtain the destination states  (NxF2)
                                             states_dest = getattr(self, str(dst_name) + '_state')
                                             h_dst = tf.gather(states_dest, destinations)
                                             F2 = int(self.entities_dimensions[message.destination_entity])

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
                                         update = getattr(self, dst_name + '_ff_update')
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
                                     if message.aggregation == 'sum':
                                         with tf.name_scope('combined_sum_preprocessing' + src_name) as _:
                                             m = tf.math.unsorted_segment_sum(messages, destinations,num_dst)  # m is the aggregated values. Sum together the values belonging to the same path
                                             setattr(self, str(src_name) + '_sum_combined', m)


                                     #combination aggreagation
                                     elif message.aggregation == 'combination':
                                         with tf.name_scope('combination_preprocessing' + src_name) as _:

                                             seq = input['seq_' + src_name + '_' + dst_name]

                                             ids = tf.stack([destinations, seq],axis=1)

                                             lens = tf.math.segment_sum(data=tf.ones_like(destinations),segment_ids=destinations)

                                             max_len = tf.reduce_max(seq) + 1

                                             shape = tf.stack([num_dst, max_len, int(self.entities_dimensions[src_name])])

                                             source_input = tf.scatter_nd(ids, messages, shape)

                                             setattr(self, str(src_name) + '_to_' + str(dst_name) +  '_combined_', source_input)

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
                                     if message.message_combination == 'concatenation':
                                         with tf.name_scope("concatenate_input_of_" + dst_name) as _:

                                             first = True
                                             for src_name in combine_sources:
                                                 with tf.name_scope("concat_" + src_name) as _:
                                                     #obtain the new states. Only one message from a certain entity type to each destination node
                                                     state = getattr(self, str(src_name)+'_sum_combined')

                                                     if first:
                                                         source_input = state
                                                         first = False
                                                     else:
                                                         #concatenate, so every node receives only one message
                                                         source_input = tf.concat([source_input, state], axis=1)

                                         #standard update with the aggreagated information
                                         with tf.name_scope("update_" + dst_name) as _:
                                             old_state = getattr(self, str(dst_name) + '_state')
                                             model = getattr(self, str(dst_name) + '_combined_update')
                                             new_state, _ = model(source_input, [old_state])
                                             setattr(self, str(dst_name) + '_state', new_state)


                                     #interleave combination
                                     elif message.message_combination == 'interleave':
                                         with tf.name_scope('interleave_' + dst_name) as _:

                                             first = True
                                             for src_name in combine_sources:
                                                 with tf.name_scope('add_' + src_name) as _:
                                                     len = getattr(self, 'lens_' + str(src_name))   #this is the vector with lens of each destination

                                                     s = getattr(self, str(src_name) + '_to_' + str(dst_name) +  '_combined_')  #input

                                                     #form the pattern to be used for the interleave
                                                     indices_source = input["indices_" + src_name + '_to_' + dst_name]
                                                     if first:
                                                         first = False
                                                         sources_input = s  # sources x max of dest x dim source
                                                         indices = indices_source
                                                         final_len = len
                                                     else:
                                                         sources_input = tf.concat([sources_input, s], axis = 1)
                                                         indices = tf.stack([indices, indices_source], axis = 0)
                                                         final_len = tf.math.add(final_len, len)    #check this len? We just sum the max of each of them?

                                             #place each of the messages into the right spot in the sequence defined
                                             with tf.name_scope('order_sources_from_' + dst_name) as _:
                                                 sources_input = tf.transpose(sources_input, perm =[1,0,2]) #destinations x length x dim_source ->  (length x destinations x dimension_source)
                                                 indices = tf.reshape(indices, [-1, 1])

                                                 sources_input = tf.scatter_nd(indices, sources_input, tf.shape(sources_input, out_type=tf.int64))

                                                 sources_input = tf.transpose(sources_input, perm=[1, 0, 2])    #destinations x length x dim_source


                                             #update with the obtained messages forming the desired sequence
                                             with tf.name_scope('update_' + dst_name) as _:
                                                 old_state = getattr(self, str(dst_name) + '_state')
                                                 model = getattr(self, str(dst_name) + '_combined_update')

                                                 gru_rnn = tf.keras.layers.RNN(model, return_sequences=True,return_state=True)

                                                 outputs, new_state = gru_rnn(inputs=sources_input, initial_state=old_state, mask=tf.sequence_mask(final_len))

                                                 setattr(self, str(dst_name) + '_state', new_state)



        #perform the predictions
        first = True
        with tf.name_scope('readout_predictions') as _:
            outputs = model_info.get_output_models()
            model_counter = 0
            #for each of the readouts defined
            for output in outputs:
                model_counter += 1

                dst_name = output.entity
                var_name = dst_name + '_' + str(model_counter)
                model = getattr(self, 'output_' + str(model_counter) +'_'+str(var_name))

                input = getattr(self, dst_name + '_state')


                # if the output we want to treat is graph, we need to aggregate all the inputs into a
                # single value to make graph predictions. By default assume we sum all of them together.
                if output.type == 'graph_prediction':
                    input = tf.reduce_sum(input, 0)
                    input = tf.reshape(input, [-1] + [input.shape.as_list()[0]])

                r = model(input)

                #concatenate all the predictions of different readouts together. MAYBE PERFORM HERE THE DENORMALIZATION?
                with tf.name_scope('Concatenate_' + output.output_label) as _:
                    if first:
                        predictions = r
                        first = False

                    else:
                        predictions = tf.concat([predictions, r], axis = 0, name="Add_output_"+ str(model_counter))

        return predictions



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

