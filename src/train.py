import tensorflow as tf
import numpy as np
from model import DSKReG
from time import time
from tqdm import tqdm
from batch_test import test, test_score

def train(args, data_loader, show_loss, show_topk=True):
    n_user, n_item, n_entity, n_relation = data_loader.n_users, data_loader.n_items, data_loader.n_entity, data_loader.n_relation

    user_item_train_dict, item_user_train_dict, user_item_test_dict, item_user_test_dict = data_loader.user_item_train_dict, data_loader.item_user_train_dict, data_loader.user_item_test_dict, data_loader.item_user_test_dict

    adj_item = data_loader.adj_item
    adj_relation = data_loader.adj_relation

    adj_user = data_loader.adj_user_np
    adj_user_relation = data_loader.adj_user_relation_np


    print("start training")
    model = DSKReG(args, n_user, n_entity, n_relation, adj_item, adj_relation, user_item_train_dict, item_user_train_dict, adj_user, adj_user_relation)
    print("model builted")

    # top-K evaluation settings

    user_list, item_set, k_list = topk_settings(show_topk, user_item_train_dict, user_item_test_dict, n_item)

    print('number of test users')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("start runing session")

        for step in range(args.n_epochs):


            """train stage"""
            t1 = time()
            loss = 0
            n_batch = data_loader.n_train//args.batch_size + 1

            for idx in tqdm(range(n_batch)):
                btime = time()
                train_feed_dict = data_loader.sample(model)
                _, batch_loss = model.train(sess, train_feed_dict)
                loss += batch_loss


            """test stage using batch_test_score"""

            ret = test_score(sess, model, user_list, item_set, user_item_train_dict, args.batch_size)

            print('precision: ', end='')
            for i in ret['precision']:
                print('%.4f\t' % i, end='')
            print()
            print('recall: ', end='')
            for i in ret['recall']:
                print('%.4f\t' % i, end='')
            print()
            print('ndgc: ', end='')
            for i in ret['ndcg']:
                print('%.4f\t' % i, end='')
            print('\n')



def topk_settings(show_topk, train_record, test_record, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        user_list = list(set(test_record.keys()))
        item_set = set(list(range(n_item)))
        return user_list, item_set, k_list
    else:
        return [None] * 5


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in tqdm(user_list):
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.pos_item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.pos_item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(set(test_record[user])))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]


    return precision, recall


def topk_eval_all(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in tqdm(user_list):
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        start = 0
        items, scores = model.get_scores_all(sess, {model.user_indices: [user],
                                                    model.pos_item_indices: test_item_list})
        print(scores)
        for item, score in zip(items, scores):
            item_score_map[item] = score


        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(set(test_record[user])))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]


    return precision, recall

