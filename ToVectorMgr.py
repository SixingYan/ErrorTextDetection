'''
在这里批量训练词向量
'''
import pickle
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd


def savePickle(path, var):
    """"""
    with open(path, 'wb') as f:
        pickle.dump(var, f)


def getDoc2Vec(args):
    _doc2vec(args)


def getDoc2Vec(args: object):
    """"""
    train = pd.read_csv(os.path.join(const.DATAPATH, 'data_train.csv'))
    texts = [list(p.split()) for p in df['sent_{}'.format(args.dtype)].values.tolist()]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=3)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    mat_train = np.matrix([model.infer_vector(doc) for doc in texts])
    savePickle(os.path.join(const.MODELPATH, 'data_train_{}_{}.vec'.format(args.dtype, args.vec)), mat_train)

    test = pd.read_csv(os.path.join(const.DATAPATH, 'data_test.csv'))
    texts = [list(p.split()) for p in df['sent_{}'.format(args.dtype)].values.tolist()]
    mat_test = np.matrix([model.infer_vector(doc) for doc in texts])
    savePickle(os.path.join(const.MODELPATH, 'data_test_{}_{}.vec'.format(args.dtype, args.vec)), mat_train)


if __name__ == '__main__':
    getDoc2Vec(args)
