import tensorflow as tf
from abc import abstractmethod
import numpy as np


EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny
LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]

############################################3
# From https://github.com/ermongroup/subsets.git
##########################################
def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random_uniform(
        tf.shape(w),
        minval=EPSILON,
        maxval=1.0)
    z = tf.log(-tf.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, separate=False):
    khot_list = []
    onehot_approx = tf.zeros_like(w, dtype=tf.float32)
    for i in range(k):
        khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        w += tf.log(khot_mask)
        onehot_approx = tf.nn.softmax(w / t, axis=-1)
        khot_list.append(onehot_approx)
    if separate:
        return khot_list
    else:
        return tf.reduce_sum(khot_list, 0)


def sample_subset(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    return continuous_topk(w, k, t)


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, K):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks,K)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, K):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings, K):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

            self.sampling_vector = tf.get_variable(
                shape=[2 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(), name='sampling_vector')
            self.sampling_bias = tf.get_variable(
                shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer(), name='sampling_bias')

        self.threshRelu = tf.keras.layers.ThresholdedReLU(theta=0.4)

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, K):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings, K)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings, K):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)


            """sampling based aggregation"""
            inputs = tf.concat([neighbor_vectors, neighbor_relations], -1)
            outputs = tf.matmul(inputs, self.sampling_vector) + self.sampling_bias
            batch_size, _, n_neighbor = user_relation_scores_normalized.shape
            outputs = tf.reshape(outputs,[batch_size,-1,n_neighbor])
            item_relation_mask = sample_subset(outputs, k=K, t=0.1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(item_relation_mask*user_relation_scores_normalized, axis=-1)
            """end of sampling"""


            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated



class SumAggregatorUser(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):

        super(SumAggregatorUser, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

            self.sampling_vector = tf.get_variable(
                shape=[2 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(), name='sampling_vector')
            self.sampling_bias = tf.get_variable(
                shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer(), name='sampling_bias')

        self.threshRelu = tf.keras.layers.ThresholdedReLU(theta=0.4)

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,  masks, K):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings, K)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings, K):
        avg = False
        if not avg:

            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_vectors, axis=-1)
            items_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            """sampling based aggregation"""
            inputs = tf.concat([neighbor_vectors, neighbor_relations], -1)
            outputs = tf.matmul(inputs, self.sampling_vector) + self.sampling_bias
            batch_size, _, n_neighbor = items_relation_scores_normalized.shape
            outputs = tf.reshape(outputs,[batch_size,-1,n_neighbor])


            item_relation_mask = sample_subset(outputs, k=K, t=0.1)

            items_relation_scores_normalized = tf.expand_dims(item_relation_mask*items_relation_scores_normalized, axis=-1)
            """end of sampling"""


            neighbors_aggregated = tf.reduce_mean(items_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated



