import tensorflow as tf
from aggregators import SumAggregator, SumAggregatorUser

class DSKReG(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_item, adj_relation, user_item_train_dict, item_user_train_dict, adj_user, adj_user_relation):
        self._parse_args(args, adj_item, adj_relation, user_item_train_dict, item_user_train_dict, adj_user, adj_user_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_item, adj_relation, user_item_train_dict, item_user_train_dict, adj_user, adj_user_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_item = adj_item
        self.adj_relation = adj_relation
        self.neighbor_sample_size = args.neighbor_sample_size


        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.user_item_train_dict = user_item_train_dict
        self.item_user_train_dict = item_user_train_dict
        self.adj_user = adj_user
        self.adj_user_relation = adj_user_relation
        self.sample_size = args.K

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.pos_item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='pos_item_indices')
        self.neg_item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='neg_item_indices')

    def _build_model(self, n_user, n_item, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=DSKReG.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_item, self.dim], initializer=DSKReG.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=DSKReG.get_initializer(), name='relation_emb_matrix')


        # [batch_size, dim]
        user_item_neighbors, user_item_relations = self.get_user_neighbors(self.user_indices)
        self.user_embeddings, self.user_aggregators = self.aggregate_user(user_item_neighbors, user_item_relations)
        print("start aggregation")
        pos_items, pos_relations = self.get_neighbors(self.pos_item_indices)
        neg_items, neg_relations = self.get_neighbors(self.neg_item_indices)
        print("stop aggregation")

        # [batch_size, dim]
        self.pos_item_embeddings, self.pos_aggregators = self.aggregate(pos_items, pos_relations)
        self.neg_item_embeddings, self.neg_aggregators = self.aggregate(neg_items, neg_relations)

        # [batch_size]
        self.pos_scores = tf.reduce_sum(self.user_embeddings * self.pos_item_embeddings, axis=1)
        self.pos_scores_normalized = tf.sigmoid(self.pos_scores)

        self.neg_scores = tf.reduce_sum(self.user_embeddings * self.neg_item_embeddings, axis=1)
        self.neg_scores_normalized = tf.sigmoid(self.neg_scores)


        # for test stage, don;t need aggregation
        self.u_e = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.pos_i_e = tf.nn.embedding_lookup(self.entity_emb_matrix, self.pos_item_indices)
        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)
        self.batch_predictions_normalizde = tf.sigmoid(self.batch_predictions)


    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_item, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def get_user_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):# now only for the first iteration
            neighbor_entities = tf.reshape(tf.gather(self.adj_user, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_user_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations




    # feature propagation
    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None,
                                    K = self.sample_size)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators



    # feature propagation
    def aggregate_user(self, entities, relations):
        aggregators = []  # store all aggregators
        user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, entities[0])
        item_neighbor_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, entities[1])
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregatorUser(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregatorUser(self.batch_size, self.dim)
            aggregators.append(aggregator)

            shape = [self.batch_size, -1, self.n_neighbor, self.dim]
            vector = aggregator(self_vectors=user_embeddings,
                                    neighbor_vectors=tf.reshape(item_neighbor_embeddings, shape),
                                    neighbor_relations=tf.reshape(relation_vectors[0], shape),
                                    user_embeddings=user_embeddings,
                                    masks=None,
                                    K = self.sample_size)

        res = tf.reshape(vector, [self.batch_size, self.dim])

        return res, aggregators



    def _build_train(self):
        # base loss bpr loss
        self.base_loss = tf.reduce_mean(tf.nn.softplus(-(self.pos_scores - self.neg_scores)))

        # l2 loss
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)

        for aggregator in self.pos_aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        for aggregator in self.neg_aggregators:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        for aggregator in self.user_aggregators:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def train(self, sess, feed_dict):
        op, loss = sess.run([self.optimizer, self.loss], feed_dict)
        return op, loss


    def get_scores(self, sess, feed_dict):

        return sess.run([self.pos_item_indices, self.pos_scores], feed_dict)

    def get_scores_all(self, sess, feed_dict):
        return sess.run([self.pos_item_indices, self.batch_predictions], feed_dict)

    def eval(self, sess, feed_dict):
        return sess.run(self.batch_predictions, feed_dict)
