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
from auxilary_classes import *
from functools import reduce


def set_model_info(model_description):
    """
    Parameters
    ----------
    model_description:    object
        Object with the json information model
    """

    global model_info
    model_info = model_description


def normalization(x, feature_list, output_names, output_normalizations, y=None):
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
                tf.compat.v1.logging.error(
                    'IGNNITION: The normalization function ' + str(norm_type) + ' is not defined in the main file.')
                sys.exit(1)

    # this normalization is only ready for one single output !!!
    if y != None:  # if we have labels to normalize
        n = len(output_names)

        for i in range(n):
            norm_type = output_normalizations[i]

            if str(norm_type) != 'None':
                try:
                    y = eval(norm_type)(y, output_names[i])
                except:
                    tf.compat.v1.logging.error(
                        'IGNNITION: The normalization function ' + str(norm_type) + ' is not defined in the main file.')
                    sys.exit(1)
        return x, y

    return x


def input_fn(data_dir, shuffle=False, training=True):
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
        output_names, output_normalizations, _ = model_info.get_output_info()
        additional_input = model_info.get_additional_input_names()
        unique_additional_input = [a for a in additional_input if a not in feature_list]

        types, shapes = {}, {}
        feature_names = []

        for a in unique_additional_input:
            types[a] = tf.int64
            shapes[a] = tf.TensorShape(None)

        for f in feature_list:
            f_name = f.name
            feature_names.append(f_name)
            types[f_name] = tf.float32
            shapes[f_name] = tf.TensorShape(None)

        for a in adjecency_info:
            types['src_' + a[0]] = tf.int64
            shapes['src_' + a[0]] = tf.TensorShape([None])
            types['dst_' + a[0]] = tf.int64
            shapes['dst_' + a[0]] = tf.TensorShape([None])
            types['seq_' + a[1] + '_' + a[2]] = tf.int64
            shapes['seq_' + a[1] + '_' + a[2]] = tf.TensorShape([None])

            if a[3] == 'True':
                types['params_' + a[0]] = tf.int64
                shapes['params_' + a[0]] = tf.TensorShape(None)

        for e in entity_list:
            types['num_' + e.name] = tf.int64
            shapes['num_' + e.name] = tf.TensorShape([])

        for i in interleave_sources:
            types['indices_' + i[0] + '_to_' + i[1]] = tf.int64
            shapes['indices_' + i[0] + '_to_' + i[1]] = tf.TensorShape([None])

        if training:  # if we do training, we also expect the labels
            ds = tf.data.Dataset.from_generator(generator,
                                                (types, tf.float32),
                                                (shapes, tf.TensorShape(None)),
                                                args=(
                                                    data_dir, feature_names, output_names, adjecency_info,
                                                    interleave_list,
                                                    unique_additional_input, training, shuffle))

        else:
            ds = tf.data.Dataset.from_generator(generator,
                                                (types),
                                                (shapes),
                                                args=(
                                                    data_dir, feature_names, output_names, adjecency_info,
                                                    interleave_list,
                                                    unique_additional_input, training, shuffle))

        # ds = ds.batch(2)

        with tf.name_scope('normalization') as _:
            if training:
                ds = ds.map(lambda x, y: normalization(x, feature_list, output_names, output_normalizations, y))

            else:
                ds = ds.map(lambda x: normalization(x, feature_list, output_names, output_normalizations))

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
    predictions = tf.reshape(predictions, [-1])
    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

    m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

    return m_r_sq, update_rsq_op


class ComnetModel(tf.keras.Model):
    """
    Class that represents the final GNN

    Methods
    ----------
    call(self, input, training=False)
        Performs the GNN's action
    get_global_var_or_input(self, var_name, input)
        Obtains the global variable with var_name if exists, or the corresponding input
    save_global_variable(self, var_name, var_value)
        Save the global variable with var_name and with the corresponding value
    get_global_variable(self, var_name)
        Obtains the global variable with the corresponding var_name
    """

    def __init__(self):
        super(ComnetModel, self).__init__()
        self.dimensions = model_info.get_input_dimensions()
        self.instances_per_stage = model_info.get_mp_instances()

        with tf.name_scope('model_initializations') as _:
            for stage in self.instances_per_stage:
                for message in stage[1]:
                    dst_name = message.destination_entity

                    with tf.name_scope('message_creation_models') as _:

                        # acces each source entity of this destination
                        for src in message.source_entities:
                            operations = src.message_formation
                            src_name = src.name
                            counter = 0

                            for operation in operations:
                                if operation.type == 'feed_forward_nn':
                                    var_name = src_name + "_to_" + dst_name + '_message_creation_' + str(counter)

                                    # Find out the dimension of the model
                                    input_nn = operation.input
                                    input_dim = 0
                                    for i in input_nn:
                                        if i == 'hs_source':
                                            input_dim += int(self.dimensions[src_name])
                                        elif i == 'hs_dest':
                                            input_dim += int(self.dimensions[dst_name])
                                        elif i == 'edge_params':
                                            input_dim += int(src.extra_parameters)  # size of the extra parameter
                                        else:
                                            dimension = getattr(self, i + '_dim')
                                            input_dim += dimension

                                    model, output_shape = operation.model.construct_tf_model(var_name, input_dim)

                                    self.save_global_variable(var_name, model)

                                    # Need to keep track of the output dimension of this one, in case we need it for a new model
                                    if operation.output_name != 'None':
                                        self.save_global_variable(operation.output_name + '_dim', output_shape)

                                    self.save_global_variable("final_message_dim_" + src.adj_vector, output_shape)

                                # if the operation is direct assignation, then the shape doesn't change
                                else:
                                    self.save_global_variable("final_message_dim_" + src.adj_vector, int(self.dimensions[src_name]))

                            counter += 1

                    # -----------------------------
                    # Creation of the update models
                    with tf.name_scope('update_models') as _:
                        update_model = message.update

                        # ------------------------------
                        # create the recurrent update models
                        if update_model.type == "recurrent_nn":
                            recurrent_cell = update_model.model
                            try:
                                recurrent_instance = recurrent_cell.get_tensorflow_object(self.dimensions[dst_name])
                                self.save_global_variable(dst_name + '_update', recurrent_instance)
                            except:
                                tf.compat.v1.logging.error(
                                    'IGNNITION: The definition of the recurrent cell in message passsing from ' + message.source_entity + ' to ' + message.destination_entity +
                                    ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                sys.exit(1)


                        # ----------------------------------
                        # create the feed-forward upddate models
                        # This only makes sense with aggregation functions that preserve one single input (not sequence)
                        else:
                            model = update_model.model
                            source_entities = message.source_entities
                            var_name = dst_name + "_ff_update"

                            with tf.name_scope(dst_name + '_ff_update') as _:
                                dst_dim = int(self.dimensions[dst_name])

                                # calculate the message dimensionality (considering that they all have the same dim)
                                # o/w, they are not combinable
                                message_dimensionality = self.get_global_variable("final_message_dim_" + source_entities[0].adj_vector)

                                # if we are concatenating by message
                                if mp.message_combination == 'concat' and mp.concat_axis == 2:
                                    message_dimensionality = reduce(lambda accum, s: accum + int(getattr(self, "final_message_dim_" + s.adj_vector)),
                                                                    source_entities, 0)

                                input_dim = message_dimensionality + dst_dim    #we will concatenate the sources and destinations

                                model, _ = model.construct_tf_model(var_name, input_dim, dst_dim, dst_name = dst_name)
                                self.save_global_variable(var_name, model)


            # --------------------------------
            # Create the readout model
            readout_operations = model_info.get_readout_operations()
            counter = 0
            for operation in readout_operations:
                if operation.type == "predict" or operation.type == 'neural_network':
                    with tf.name_scope("readout_architecture"):
                        input_dim = reduce(lambda accum, i: accum + int(self.dimensions[i]), operation.input, 0)
                        #model = operation.architecture.construct_tf_readout(counter, input_dim)
                        model, _ = operation.architecture.construct_tf_model('readout_model' + str(counter), input_dim, is_readout = True)
                        setattr(self, 'readout_model_' + str(counter), model)

                    # save the dimensions of the output
                    if operation.type == 'neural_network':
                        self.dimensions[operation.output_name] = model.layers[-1].output.shape[1]


                elif operation.type == 'pooling':
                    dimensionality = self.dimensions[operation.input[0]]

                    # add the new dimensionality to the input_dimensions tensor
                    self.dimensions[operation.output_name] = dimensionality

                elif operation.type == 'product':
                    if operation.type_product == 'element_wise':
                        self.dimensions[operation.output_name] = self.dimensions[operation.input[0]]

                    elif operation.type_product == 'dot_product':
                        self.dimensions[operation.output_name] = 1

                elif operation.type == 'extend_adjacencies':
                    self.dimensions[operation.output_name[0]] = self.dimensions[operation.input[0]]
                    self.dimensions[operation.output_name[1]] = self.dimensions[operation.input[1]]

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

        # Initialize all the hidden states for all the nodes.
        with tf.name_scope('hidden_states') as _:
            for entity in entities:
                with tf.name_scope('hidden_state_' + str(entity.name)) as _:
                    state = entity.calculate_hs(input)
                    self.save_global_variable(str(entity.name) + '_state', state)

        # -----------------------------------------------------------------------------------
        # MESSAGE PASSING PHASE
        with tf.name_scope('message_passing') as _:

            for j in range(model_info.get_mp_iterations()):

                with tf.name_scope('iteration_' + str(j)) as _:

                    for stage in self.instances_per_stage:
                        step_name = stage[0]

                        # given one message from a given step
                        for mp in stage[1]:
                            dst_name = mp.destination_entity
                            dst_states = getattr(self, str(dst_name) + '_state')
                            num_dst = input['num_' + dst_name]

                            #with tf.name_scope('mp_to_' + dst_name + 's') as _:
                            with tf.name_scope(mp.source_entities[0].name + 's_to_' + dst_name + 's') as _:
                                first_src = True
                                with tf.name_scope('message') as _:
                                    for src in mp.source_entities:
                                        src_name = src.name

                                        # prepare the information
                                        src_idx, dst_idx, seq = input['src_' + src.adj_vector], input['dst_' + src.adj_vector], input['seq_' + src_name + '_' + dst_name]
                                        src_states = self.get_global_variable(str(src_name) + '_state')

                                        with tf.name_scope('create_message_' + src_name + '_to_' + dst_name) as _:
                                            src_messages = tf.gather(src_states, src_idx)
                                            dst_messages = tf.gather(dst_states, dst_idx)
                                            message_creation_models = src.message_formation

                                            #by default, the source hs are the messages
                                            final_messages = src_messages

                                            counter = 0
                                            for model in message_creation_models:
                                                if model.type == 'feed_forward_nn':
                                                    with tf.name_scope('apply_nn_' + str(counter)) as _:
                                                        var_name = src_name + "_to_" + dst_name + '_message_creation_' + str(
                                                            counter)  # careful. This name could overlap with another model
                                                        message_creator = getattr(self, var_name)
                                                        first = True
                                                        with tf.name_scope('create_input') as _:
                                                            for i in model.input:
                                                                if i == 'hs_source':
                                                                    new_input = src_messages
                                                                elif i == 'hs_dest':
                                                                    new_input = dst_messages
                                                                elif i == 'edge_params':
                                                                    new_input = tf.cast(
                                                                        input['params_' + src.adj_vector],
                                                                        tf.float32)
                                                                else:
                                                                    new_input = getattr(self, i + '_var')

                                                                # accumulate the results
                                                                if first:
                                                                    first = False
                                                                    input_nn = new_input
                                                                else:
                                                                    input_nn = tf.concat([input_nn, new_input], axis=1)

                                                        with tf.name_scope('create_message') as _:
                                                            result = message_creator(input_nn)
                                                            if model.output_name != 'None':
                                                                self.save_global_variable(model.output_name + 'var', result)

                                                            final_messages = result  # by default, the message is always the result from the last operation

                                                counter += 1


                                        with tf.name_scope('combine_messages_' + src_name + '_to_' + dst_name) as _:
                                            # prepare the input for the concat
                                            ids = tf.stack([dst_idx, seq], axis=1)

                                            lens = tf.math.unsorted_segment_sum(tf.ones_like(dst_idx), dst_idx, num_dst)    #CHECK

                                            max_len = tf.reduce_max(seq) + 1

                                            shape = tf.stack([num_dst, max_len, int(getattr(self, "final_message_dim_" + src.adj_vector))])

                                            s = tf.scatter_nd(ids, final_messages, shape)  # find the input ordering it by sequence

                                            # shape: num_dst x srcs for destination x dim(src)
                                            s = tf.RaggedTensor.from_tensor(s,lens)

                                            aggr = mp.aggregation

                                            #now we concatenate them with the already calculated messages (considering the aggregation)
                                            if isinstance(aggr, Concat_aggr):
                                                with tf.name_scope("concat_" + src_name) as _:
                                                    if first_src:
                                                        src_input = s
                                                        final_len = lens
                                                        first_src = False
                                                    else:
                                                        src_input = tf.concat([src_input, s], axis=aggr.concat_axis)
                                                        if aggr.concat_axis == 1:  # if axis=2, then the number of messages received is the same. Simply create bigger messages
                                                            final_len += lens

                                            elif isinstance(aggr, Interleave_aggr):
                                                with tf.name_scope('add_' + src_name) as _:
                                                    indices_source = input["indices_" + src_name + '_to_' + dst_name]
                                                    if first_src:
                                                        first_src = False
                                                        src_input = s  # destinations x max_of_sources_to_dest x dim_source
                                                        indices = indices_source
                                                        final_len = lens
                                                    else:
                                                        # destinations x max_of_sources_to_dest_concat x dim_source
                                                        src_input = tf.concat([src_input, s], axis=1)
                                                        indices = tf.stack([indices, indices_source], axis=0)
                                                        final_len = tf.math.add(final_len, lens)


                                            # if we must aggregate them together into a single embedding (sum, attention, ordered)
                                            else:
                                                # obtain the overall input of each of the destinations
                                                if first_src:
                                                    first_src = False
                                                    src_input = s  # destinations x sources_to_dest x dim_source
                                                    comb_src_states, comb_dst_idx, comb_seq = src_messages, dst_idx, seq  #we need this for the attention and convolutional mechanism
                                                    final_len = lens

                                                    # dimension of one source (all must be the same)
                                                    F1 = int(self.dimensions[src_name])

                                                else:
                                                    # destinations x max_of_sources_to_dest_concat x dim_source
                                                    src_input = tf.concat([src_input, s], axis=1)
                                                    comb_src_states = tf.concat([comb_src_states, src_messages], axis=0)
                                                    comb_dst_idx = tf.concat([comb_dst_idx, dst_idx], axis=0)

                                                    # lens of each src-dst value
                                                    aux_lens = tf.gather(lens, dst_idx)
                                                    aux_seq = seq + aux_lens  # sum to the sequences the current length for each dest
                                                    comb_seq = tf.concat([comb_seq, aux_seq], axis=0)

                                                    final_len = tf.math.add(final_len, lens)


                                # transform it back to tensor
                                src_input = src_input.to_tensor()

                                # --------------
                                # perform the actual aggregation
                                aggr = mp.aggregation

                                #if ordered, we dont need to do anything. Already in the right shape

                                # sum aggregation
                                with tf.name_scope('aggregation') as _:
                                    if aggr.type == 'sum':
                                        src_input = aggr.calculate_input(src_input)

                                    # attention aggreagation
                                    elif aggr.type == 'attention':
                                        F2 = int(self.dimensions[mp.destination_entity])
                                        kernel1 = self.add_weight(shape=(F1, F1))  
                                        kernel2 = self.add_weight(shape=(F2, F1))
                                        attn_kernel = self.add_weight(shape=(2 * F1, 1))
                                        #check F1, if we use a function that changes the message's size.

                                        src_input = aggr.calculate_input(comb_src_states, comb_dst_idx, dst_states, comb_seq, num_dst, kernel1, kernel2, attn_kernel)

                                    # convolutional aggregation
                                    elif aggr.type == 'convolutional':
                                        print("Here we would do a convolution")


                                    elif aggr.type == 'interleave':
                                        # place each of the messages into the right spot in the sequence defined
                                        src_input = aggr.calculate_input(src_input, indices)



                                # ---------------------------------------
                                # update
                                with tf.name_scope('update') as _:
                                    update_model = mp.update
                                    old_state = self.get_global_variable(str(dst_name) + '_state')

                                    # recurrent update
                                    if update_model.type == "recurrent_nn":
                                        model = self.get_global_variable(str(dst_name) + '_update')

                                        if aggr.type == 'sum' or aggr.type == 'attention' or aggr.type == 'convolutional':
                                            new_state = update_model.model.perform_unsorted_update(model, src_input, old_state)

                                        # if the aggregation was ordered
                                        else:
                                            new_state = update_model.model.perform_sorted_update(model,src_input, dst_name, old_state, final_len, num_dst)

                                        self.save_global_variable(str(dst_name) + '_state', new_state)


                                    # feed-forward update:
                                    # restriction: It can only be used if the aggreagation was not ordered.
                                    elif update_model.type == 'feed_forward_nn':
                                        var_name = dst_name + "_ff_update"
                                        update = self.get_global_variable(var_name)

                                        # now we need to obtain for each adjacency the concatenation of the source and the destination
                                        update_input = tf.concat([src_input, old_state], axis=1)
                                        new_state = update(update_input)
                                        self.save_global_variable(str(dst_name) + '_state', new_state)  #update the old state


        # -----------------------------------------------------------------------------------
        # READOUT PHASE
        with tf.name_scope('readout_predictions') as _:
            readout_opeartions = model_info.get_readout_operations()

            counter = 0
            for operation in readout_opeartions:
                if operation.type == 'neural_network' or operation.type == 'predict':

                    first = True
                    for i in operation.input:
                        new_input = self.get_global_var_or_input(i, input)

                        if first:
                            nn_input = new_input
                            first = False
                        else:
                            nn_input = tf.concat([nn_input, new_input], axis=1)

                    readout_nn = self.get_global_variable('readout_model_' + str(counter))
                    result = readout_nn(nn_input, training = training)

                    if operation.type == 'neural_network':
                        self.save_global_variable(operation.output_name + '_state', result)
                    else:
                        return result

                elif operation.type == "pooling":
                    # obtain the input of the pooling operation
                    pooling_input = self.get_global_var_or_input(operation.input[0], input)

                    result = operation.calculate(pooling_input)
                    self.save_global_variable(operation.output_name + '_state', result)


                elif operation.type == 'product':
                    product_input1 = self.get_global_var_or_input(operation.input[0], input)
                    product_input2 = self.get_global_var_or_input(operation.input[1], input)

                    result = operation.calculate(product_input1, product_input2)
                    self.save_global_variable(operation.output_name + '_state', result)

                elif operation.type == 'extend_adjacencies':
                    adj_src = input['src_' + operation.adj_list]
                    adj_dst = input['dst_' + operation.adj_list]

                    src_states = self.get_global_var_or_input(operation.input[0], input)
                    dst_states = self.get_global_var_or_input(operation.input[1], input)

                    extended_src, extended_dst = operation.calculate(src_states, adj_src, dst_states, adj_dst)
                    self.save_global_variable(operation.output_name[0] + '_state', extended_src)
                    self.save_global_variable(operation.output_name[1] + '_state', extended_dst)

                counter += 1


    def get_global_var_or_input(self, var_name, input):
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

        try:
            return self.get_global_variable(var_name + '_state')
        except:
            return input[var_name]



    # creates a new global variable
    def save_global_variable(self, var_name, var_value):
        """
        Parameters
        ----------
        var_name:    String
            Name of the global variable to save
        var_value:    tensor
            Tensor value of the new global variable
        """
        setattr(self, var_name, var_value)


    def get_global_variable(self, var_name):
        """
        Parameters
        ----------
        var_name:    String
            Name of the global variable to save
        """
        return getattr(self, var_name)



def model_fn(features, labels, mode):
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

    # create the model
    model = ComnetModel()

    # peform the predictions
    predictions = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # prediction mode. Denormalization is done if so specified
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_names, _, output_denorm = model_info.get_output_info()  # for now suppose we only have one output type

        try:
            predictions = eval(output_denorm[0])(predictions, output_names[0])
        except:
            tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_names[
                0] + ' was not defined. The output will be normalized.')

        return tf.estimator.EstimatorSpec(
            mode, predictions={
                'predictions': predictions
            })

    # dynamically define the loss function from the keras documentation
    name = model_info.get_loss()
    loss = globals()[name]
    loss_function = loss()

    regularization_loss = sum(model.losses)

    loss = loss_function(labels, predictions)

    total_loss = loss + regularization_loss
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        # perform denormalization if defined
        output_names, _, output_denorm = model_info.get_output_info()
        try:
            labels = eval(output_denorm[0])(labels, output_names[0])
            predictions = eval(output_denorm[0])(predictions, output_names[0])
        except:
            tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_names[
                0] + ' was not defined. The output (and statistics) will use the normalized values.')

        # metrics calculations
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

    # dynamically define the optimizer
    optimizer_params = model_info.get_optimizer()
    op_type = optimizer_params['type']
    del optimizer_params['type']

    # dynamically define the adaptative learning rate if needed

    if 'schedule' in optimizer_params:
        schedule = optimizer_params['schedule']
        type = schedule['type']
        del schedule['type']  # so that only the parameters remain
        s = globals()[type]
        # create an instance of the schedule class indicated by the user. Accepts any schedule from keras documentation
        optimizer_params['learning_rate'] = s(**schedule)
        del optimizer_params['schedule']

    #create the optimizer
    o = globals()[op_type]
    optimizer = o(
        **optimizer_params)  # create an instance of the optimizer class indicated by the user. Accepts any loss function from keras documentation

    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

    logging_hook = tf.estimator.LoggingTensorHook(
        {"Loss": loss,
         "Regularization loss": regularization_loss,
         "Total loss": total_loss,
         "Prediction": predictions}
        , every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )
