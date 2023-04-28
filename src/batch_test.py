
import multiprocessing
import metrics
import heapq
import numpy as np
import argparse
from data_loader import Dataloader
from tqdm import tqdm
parser = argparse.ArgumentParser()


# # movie
# parser.add_argument('--dataset', type=str, default='movie_1m', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=500, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
# parser.add_argument('--K', type=int, default=8, help='number of neighbors after sample')

# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=500, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--K', type=int, default=8, help='number of neighbors after sample')


# # # lastfm
# parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=500, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--K', type=int, default=8, help='number of neighbors after sample')

#  # music
# parser.add_argument('--dataset', type=str, default='music_small', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=500, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--K', type=int, default=8, help='number of neighbors after sample')



args = parser.parse_args()
if args.dataset == 'movie_1m':
    metapath = {0: [0, 0], 1: [2, 2], 2: [4, 4]}
elif args.dataset == 'music':

    metapath = {0:[1,1],1:[2,2], 2:[6,6], 3:[8,8]} #

elif args.dataset =='music_small':
    metapath = {0: [0,0], 1:[9,9], 2:[12,12], 3:[26,26], 4:[31,31], 5:[35,35], 6:[17,17], 7:[26,26], 8:[52,52] }

elif args.dataset == 'amazon-book':
    metapath = {0:[10,10],1:[15,15],2:[16,16],3:[18,18],4:[20,20]}

elif args.dataset == 'yelp2018':
    metapath={0:[24,24],1:[3,3],2:[11,11],3:[19,19],4:[30,30],5:[2,2],6:[14,14],7:[8,8],8:[6,6],9:[17,17], 10:[9,9],11:[13,13], 12:[15,15], 13:[32,32],14:[40,40]}

elif args.dataset == 'book':
    metapath = {0:[0,0], 1:[1,1], 2:[20,20]}

path = "../data/" +args.dataset
dataloader = Dataloader(path, args, metapath)
data_generator = dataloader


cores = multiprocessing.cpu_count() // 2
Ks = [1, 2, 5, 10, 20, 50, 100]

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size



############################################3
# From https://github.com/xiangwang1223/knowledge_graph_attention_network.git
##########################################
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.user_item_train_dict[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.user_item_test_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE *2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                                   user_batch=user_batch,
                                                                   item_batch=item_batch)
                i_rate_batch = model.eval(sess, feed_dict=feed_dict)
                i_rate_batch = i_rate_batch.reshape((-1, len(item_batch)))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
            feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                               user_batch=user_batch,
                                                               item_batch=item_batch)
            rate_batch = model.eval(sess, feed_dict=feed_dict)
            rate_batch = rate_batch.reshape((-1, len(item_batch)))

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result



def test_score(sess, model, user_list, item_set, train_record, batch_size):
    # use the pos_item_score for the ranking prediction
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    n_test_users = len(user_list)
    for user in tqdm(user_list):
        test_item_list = list(item_set)
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

        item_score = [item_score_map[item] for item in test_item_list]
        re = test_one_user((item_score, user))
        result['precision'] += re['precision'] / n_test_users
        result['recall'] += re['recall'] / n_test_users
        result['ndcg'] += re['ndcg'] / n_test_users
        result['hit_ratio'] += re['hit_ratio'] / n_test_users
        result['auc'] += re['auc'] / n_test_users

    return result







