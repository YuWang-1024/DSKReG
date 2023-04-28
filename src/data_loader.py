import numpy as np
import os
import random as rd

TEST_THRESHOLD = {'music_small': 2, 'movie_1m': 140, 'book': 14}


class Dataloader(object):
    """
    load the train test from fixed train.txt and test.txt file
    sample train batch
    """
    def __init__(self, path, args, metapathes):
        print(path)
        self.path = path
        self.batch_size = args.batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        item_file = path + '/item_list.txt'
        self.max_length = 500

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 1, 1

        self.user_item_train_dict = {} # store the user:[items] for training
        self.item_user_train_dict = {} # store the item:[users] for the training

        with open(train_file) as f:

            for l in f.readlines():

                l = l.strip('\n').split(' ')
                if len(l) > 0:
                    uid = int(l[0])
                    items = [int(i) for i in l[1:]]
                    self.n_train += len(items)
                    if uid not in self.user_item_train_dict:
                        self.user_item_train_dict[uid] = list(set(items))
                    for item in items:
                        if item not in self.item_user_train_dict:
                            self.item_user_train_dict[item] = []
                        self.item_user_train_dict[item].append(uid)

        self.user_item_test_dict = {}
        self.item_user_test_dict = {}

        with open(test_file) as f:
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                if len(l)>0:
                    uid = int(l[0])
                    items = [int(i) for i in l[1:]] # there is user that don't have pos test items
                    self.n_test += len(items)
                    if uid not in self.user_item_test_dict:
                        self.user_item_test_dict[uid] = list(set(items))
                    for item in items:
                        if item not in self.item_user_test_dict:
                            self.item_user_test_dict[item] = []
                        self.item_user_test_dict[item].append(uid)

        self.n_users = len(self.user_item_train_dict.keys())
        # self.n_items = len(set(self.item_user_train_dict.keys()) | set(self.item_user_test_dict.keys()))
        with open(item_file) as f:
            for l in f.readlines():
                self.n_items += 1




        """load adj items and relations from the kg"""
        self.n_entity, self.n_relation, self.adj_item, self.adj_relation = load_kg_for_SKGNN(args, metapathes,
                                                                                             self.n_items)

        self.n_relation = self.n_relation + 1  # the last relation represents the user-item pair relation

        """load adj items and relations from ratings"""

        if os.path.exists(path + '/adj_user.npy'):
            self.adj_user_np = np.load(path + '/adj_user.npy')
            self.adj_user_relation_np = np.load(path + '/adj_user_relation.npy')
        else:
            self.adj_user_np = np.full((self.n_users, self.max_length),
                                       -1)  # store the rated items from of specific user with fixed size
            self.adj_user_relation_np = np.full((self.n_users, self.max_length), -1)

            for u, items in self.user_item_train_dict.items():
                if len(items) < self.max_length:
                    sample_idx = np.random.choice(list(range(len(items))), size=self.max_length, replace=True)
                    self.adj_user_np[u] = np.array([items[i] for i in sample_idx]).astype(int)
                    self.adj_user_relation_np[u] = np.array([self.n_relation-1 for i in range(self.max_length)])
                else:
                    sample_idx = np.random.choice(list(range(len(items))), size=self.max_length, replace=False)
                    self.adj_user_np[u] = np.array([items[i] for i in sample_idx]).astype(int)
                    self.adj_user_relation_np[u] = np.array([self.n_relation-1 for i in range(self.max_length)])
            np.save(path + '/adj_user.npy', self.adj_user_np)
            np.save(path + '/adj_user_relation.npy', self.adj_user_relation_np)

        print(f"number of users: {self.n_users}, min_indices: {min(self.user_item_train_dict.keys())}, max_indices: {max(self.user_item_train_dict.keys())}")
        print(f"number of items: {self.n_items}")
        print(f"number of entity: {self.n_entity}")
        print(f"number of relations: {self.n_relation}")




    def sample(self, model):
        if self.batch_size <= self.n_users:
            users = np.random.choice(list(range(self.n_users)), size=self.batch_size, replace=False)
        else:
            users = [rd.choice(self.n_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.user_item_train_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.user_item_train_dict[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        feed_dict = {
            model.user_indices: users,
            model.pos_item_indices: pos_items,
            model.neg_item_indices: neg_items
        }

        return feed_dict


    def generate_test_feed_dict(self, model, user_batch, item_batch):

        feed_dict ={
            model.user_indices: user_batch,
            model.pos_item_indices: item_batch,

        }

        return feed_dict


"""load kg and rating file for skgnn"""


def load_kg_for_SKGNN(args, metapathes, n_item):
    """
    construct adj_item and adj_relation from knowledge graph along the metapathes
    :param args:
    :param metapathes:
    :param n_item:
    :return:
    """
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(metapathes.keys()) + 1 # the last represent the

    adj_file = '../data/' + args.dataset + '/adj_'
    if os.path.exists(adj_file + 'item.npy'):
        print("load local adjacent information from the KG")
        adj_item = np.load(adj_file + 'item.npy')
        adj_relation = np.load(adj_file + 'relation.npy')
    else:
        kg, adj_item, adj_relation = construct_kg_forSKGNN(kg_np, metapathes, n_item, n_relation)
        np.save(adj_file + 'item.npy', adj_item)
        np.save(adj_file + 'relation.npy', adj_relation)

    return n_entity, n_relation, adj_item, adj_relation

def construct_kg_forSKGNN(kg_np, metapathes, n_item, n_relation):
    print('constructing knowledge graph and find all metapath neighbors')
    kg = dict()
    relation_head_tail = dict()
    head_relation_tail = dict()
    tail_relation_head = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

        if relation not in relation_head_tail:
            relation_head_tail[relation] = {}
        head_tail_dict = relation_head_tail[relation]
        if head not in head_tail_dict:
            head_tail_dict[head] = []
        head_tail_dict[head].append(tail)

        if head not in head_relation_tail:
            head_relation_tail[head] = {}
        relation_tail_dict = head_relation_tail[head]
        if relation not in relation_tail_dict:
            relation_tail_dict[relation] = []
        relation_tail_dict[relation].append(tail)

        if tail not in tail_relation_head:
            tail_relation_head[tail] = {}
        relation_head_dict = tail_relation_head[tail]
        if relation not in relation_head_dict:
            relation_head_dict[relation] = []
        relation_head_dict[relation].append(head)

    print('finish loading KG')

    print(f"number of heads {len(head_relation_tail.keys())}")
    print(f"number of relations: {len(relation_head_tail.keys())}")
    print(f"number of tail: {len(tail_relation_head.keys())}")


    print('start constructing adj_item and adj_relation')

    """ find metapath neighbor items and store corresponding relation """
    adj_item = {}
    adj_relation = {}
    for head in range(n_item):
        # for every item, we construct an adj list of all meta path
        neighbor = np.asarray([])
        relation = np.asarray([])

        if head in head_relation_tail:
            relation_tail_dict = head_relation_tail[head]
        else:
            relation_tail_dict = {r:[head] for r in range(n_relation)}
            head_relation_tail[head] = relation_tail_dict
        relation_tail = relation_tail_dict

        for r, [r1, r2] in metapathes.items():
            if r1 in relation_tail:
                tails = relation_tail[r1]
                for t in tails:
                    if t < n_item:
                        neighbor = np.concatenate([neighbor, [t]])
                        relation = np.concatenate([relation, [r]])
                    if t in tail_relation_head:
                        relation_head = tail_relation_head[t]
                        if r2 in relation_head:
                            temp_neighbor = np.asarray(relation_head[r2])
                            # remove head item to avoid self loops
                            temp_neighbor = temp_neighbor[temp_neighbor != head]
                            #remove entity
                            temp_neighbor = temp_neighbor[temp_neighbor < n_item]
                            neighbor = np.concatenate([neighbor, temp_neighbor])
                            relation = np.concatenate([relation, np.full((len(temp_neighbor)), r)])
        if len(neighbor) >0:
            adj_item[head] = neighbor
            adj_relation[head] = relation
        else: # means there is no connected items along all meta-path, we set it connected to itself
            adj_item[head] = [head]
            adj_relation[head] = [len(metapathes.keys())]


    print('padding the adj to the max_length')
    max_length = 500
    print(f"max_length: {max_length}")
    adj_item_np = np.full((n_item, max_length),-1)
    adj_relation_np = np.full((n_item, max_length),-1)
    for head in range(n_item):
        neighbor = adj_item[head]
        relation = adj_relation[head]
        if len(neighbor) < max_length:
            sample_idx = np.random.choice(list(range(len(neighbor))), size=max_length, replace=True)
            adj_item_np[head] = np.array([neighbor[i] for i in sample_idx]).astype(int)
            adj_relation_np[head] = np.array([relation[i] for i in sample_idx]).astype(int)
        else:
            sample_idx = np.random.choice(list(range(len(neighbor))), size=max_length, replace=False)
            adj_item_np[head] = np.array([neighbor[i] for i in sample_idx]).astype(int)
            adj_relation_np[head] = np.array([relation[i] for i in sample_idx]).astype(int)

    return kg, adj_item_np, adj_relation_np


