import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie_1m': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music_small': 'user_artists.dat'})
SEP = dict({'movie_1m': '::', 'book': ';', 'music_small': '\t'})
THRESHOLD = dict({'movie_1m': 4, 'book': 0, 'music_small': 0})


def read_item_index_to_entity_id_file():
    """
    construct the mapping dict: item_index_old_to_new and entity_id2index
    :return:
    """
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    writer = open('../data/' + DATASET + '/item_list.txt', 'w', encoding='utf-8')
    writer.write('org_id\tremap_id\tsatori_id\n')

    entity_writer = open('../data/' + DATASET + '/entity_list.txt', 'w', encoding='utf-8')
    entity_writer.write('org_id\tremap_id\n')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
        writer.write('%s\t%d\t%s\n' %(item_index, i, satori_id))
        entity_writer.write(('%s\t%d\n') %(satori_id, i))
    writer.close()
    entity_writer.close()



def convert_rating():
    """
    rehash the user index and item index
    parse the ratings above threshold as positive rating -> positive sample, under threshold as negative rating.
    random choice same number of negative sample from (item set - postive set - negative set)
    :return:
    """
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    user_cnt = 0
    user_index_old2new = dict()
    user_train_dict = {}
    user_test_dict = {}

    user_list_writer = open('../data/' + DATASET + '/user_list.txt', 'w', encoding='utf-8')
    user_list_writer.write('org_id\tremap_id\n')

    train_list_writer = open('../data/' + DATASET + '/train.txt', 'w', encoding='utf-8')
    test_list_writer = open('../data/' + DATASET + '/test.txt', 'w', encoding='utf-8')


    n_user = len(set(user_pos_ratings.keys()))
    # random split the test user
    test_user_indices = np.random.choice(list(range(n_user)), size=int(0.2*n_user), replace=False)

    rating_cnt = 0
    train_rating_cnt = 0
    test_rating_cnt = 0

    for user_index_old, pos_item_set in user_pos_ratings.items():
        rating_cnt += len(pos_item_set)
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        if user_index in test_user_indices: # split pos items of current user into train and test
            test_ratio = 0.2
            pos_item_list = list(pos_item_set)
            test_indices = np.random.choice(list(range(len(pos_item_list))), size=int(test_ratio*len(pos_item_list)), replace=False)
            if len(test_indices)>0:
                train_indices = set(range(len(pos_item_list))) - set(test_indices)
                train_items = [pos_item_list[i] for i in train_indices]
                test_items = [pos_item_list[i] for i in test_indices]
                user_train_dict[user_index] = train_items
                user_test_dict[user_index] = test_items

                # write the train test user lines into file
                user_list_writer.write('%d\t%d\n' % (user_index_old, user_index))

                train_items_str = ' '.join(str(x) for x in train_items)
                test_items_str = ' '.join(str(x) for x in test_items)

                train_list_writer.write('%d %s\n' % (user_index, train_items_str))
                test_list_writer.write('%d %s\n' % (user_index, test_items_str))
                train_rating_cnt += len(train_items)
                test_rating_cnt += len(test_items)

            else:
                user_train_dict[user_index] = list(pos_item_set)
                user_list_writer.write('%d\t%d\n' % (user_index_old, user_index))
                train_items_str = ' '.join(str(x) for x in user_train_dict[user_index])
                train_list_writer.write('%d %s\n' % (user_index, train_items_str))
                train_rating_cnt += len(pos_item_set)
        else:
            user_train_dict[user_index] = list(pos_item_set)
            user_list_writer.write('%d\t%d\n' % (user_index_old, user_index))
            train_items_str = ' '.join(str(x) for x in user_train_dict[user_index])
            train_list_writer.write('%d %s\n' % (user_index, train_items_str))
            train_rating_cnt += len(pos_item_set)



    user_list_writer.close()
    train_list_writer.close()
    test_list_writer.close()



    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))
    print(f'number of train users: {len(user_train_dict.keys())}')
    print(f"number of test users: {len(user_test_dict.keys())}")
    print(f"number of ratings: {rating_cnt}")
    print(f"number of train ratings: {train_rating_cnt}")
    print(f"number of test ratings: {test_rating_cnt}")

def downsample_movie():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    ratings = []

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue

        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        ratings.append((user_index_old, item_index, rating))

    #down sample the rating
    selected_index = np.random.choice(list(range(len(ratings))), size=int(0.1*len(ratings)), replace=False)

    # construct user_index dict using down-sampled ratings
    for i in selected_index:
        selected_rating = ratings[i]
        user_index_old = int(selected_rating[0])
        item_index = selected_rating[1]
        rating = float(selected_rating[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    user_cnt = 0
    user_index_old2new = dict()
    user_train_dict = {}
    user_test_dict = {}

    user_list_writer = open('../data/' + DATASET + '/user_list.txt', 'w', encoding='utf-8')
    user_list_writer.write('org_id\tremap_id\n')

    train_list_writer = open('../data/' + DATASET + '/train.txt', 'w', encoding='utf-8')
    test_list_writer = open('../data/' + DATASET + '/test.txt', 'w', encoding='utf-8')


    n_user = len(set(user_pos_ratings.keys()))
    # random split the test user
    test_user_indices = np.random.choice(list(range(n_user)), size=int(0.2*n_user), replace=False)

    rating_cnt = 0
    train_rating_cnt = 0
    test_rating_cnt = 0

    for user_index_old, pos_item_set in user_pos_ratings.items():
        rating_cnt += len(pos_item_set)
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        if user_index in test_user_indices: # split pos items of current user into train and test
            test_ratio = 0.2
            pos_item_list = list(pos_item_set)
            test_indices = np.random.choice(list(range(len(pos_item_list))), size=int(test_ratio*len(pos_item_list)), replace=False)
            if len(test_indices)>0:
                train_indices = set(range(len(pos_item_list))) - set(test_indices)
                train_items = [pos_item_list[i] for i in train_indices]
                test_items = [pos_item_list[i] for i in test_indices]
                user_train_dict[user_index] = train_items
                user_test_dict[user_index] = test_items

                # write the train test user lines into file
                user_list_writer.write('%d\t%d\n' % (user_index_old, user_index))

                train_items_str = ' '.join(str(x) for x in train_items)
                test_items_str = ' '.join(str(x) for x in test_items)

                train_list_writer.write('%d %s\n' % (user_index, train_items_str))
                test_list_writer.write('%d %s\n' % (user_index, test_items_str))
                train_rating_cnt += len(train_items)
                test_rating_cnt += len(test_items)

            else:
                user_train_dict[user_index] = list(pos_item_set)
                user_list_writer.write('%d\t%d\n' % (user_index_old, user_index))
                train_items_str = ' '.join(str(x) for x in user_train_dict[user_index])
                train_list_writer.write('%d %s\n' % (user_index, train_items_str))
                train_rating_cnt += len(pos_item_set)
        else:
            user_train_dict[user_index] = list(pos_item_set)
            user_list_writer.write('%d\t%d\n' % (user_index_old, user_index))
            train_items_str = ' '.join(str(x) for x in user_train_dict[user_index])
            train_list_writer.write('%d %s\n' % (user_index, train_items_str))
            train_rating_cnt += len(pos_item_set)



    user_list_writer.close()
    train_list_writer.close()
    test_list_writer.close()



    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))
    print(f'number of train users: {len(user_train_dict.keys())}')
    print(f"number of test users: {len(user_test_dict.keys())}")
    print(f"number of ratings: {rating_cnt}")
    print(f"number of train ratings: {train_rating_cnt}")
    print(f"number of test ratings: {test_rating_cnt}")

    return 0

def convert_kg():
    """
    rehash the entity index and relation index
    :return:
    """
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    n_item = entity_cnt
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    entity_writer = open('../data/' + DATASET + '/entity_list.txt', 'a', encoding='utf-8')
    relation_writer = open('../data/' + DATASET + '/relation_list.txt', 'w', encoding='utf-8')
    relation_writer.write('org_id\tremap_id\n')

    files = []
    if DATASET == 'movie_1m':
        files.append(open('../data/' + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + DATASET + '/kg.txt', encoding='utf-8'))
    triple_cnt = 0

    if DATASET == 'movie_1m':
        for file in files:
            for line in file:
                array = line.strip().split('\t')

                head_old = array[0]
                relation_old = array[1]
                tail_old = array[2]

                if head_old in entity_id2index:
                    head = entity_id2index[head_old]
                    triple_cnt+=1
                    if tail_old not in entity_id2index:
                        entity_id2index[tail_old] = entity_cnt
                        entity_writer.write('%s\t%d\n' % (head_old, entity_cnt))
                        entity_cnt += 1
                    tail = entity_id2index[tail_old]

                    if relation_old not in relation_id2index:
                        relation_id2index[relation_old] = relation_cnt
                        relation_writer.write('%s\t%d\n' % (relation_old, relation_cnt))
                        relation_cnt += 1
                    relation = relation_id2index[relation_old]

                    writer.write('%d\t%d\t%d\n' % (head, relation, tail))


    else:
        for file in files:
            for line in file:
                array = line.strip().split('\t')

                head_old = array[0]
                relation_old = array[1]
                tail_old = array[2]

                if head_old not in entity_id2index:
                    entity_id2index[head_old] = entity_cnt
                    entity_writer.write('%s\t%d\n' % (head_old, entity_cnt))
                    entity_cnt += 1
                head = entity_id2index[head_old]

                if tail_old not in entity_id2index:
                    entity_id2index[tail_old] = entity_cnt
                    entity_writer.write('%s\t%d\n' % (head_old, entity_cnt))
                    entity_cnt += 1
                tail = entity_id2index[tail_old]

                if relation_old not in relation_id2index:
                    relation_id2index[relation_old] = relation_cnt
                    relation_writer.write('%s\t%d\n' % (relation_old, relation_cnt))
                    relation_cnt += 1
                relation = relation_id2index[relation_old]

                writer.write('%d\t%d\t%d\n' % (head, relation, tail))


    writer.close()
    entity_writer.close()
    relation_writer.close()

    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)
    print(f'number of triples: {triple_cnt}')





if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()
    read_item_index_to_entity_id_file()
    if DATASET == 'movie_1m':
        downsample_movie()
    else: convert_rating()
    # convert_rating()
    convert_kg()





