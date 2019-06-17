import os
import sys
from typing import List
import pickle
import gc
from collections import Counter

from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='Progress')

import const
from LangModelMgr import LanModel


if __name__ == '__main__':
    extract()
    # drop(featues)


def extract():
    _extract_features('data_train.csv', 'data_train_feat.csv')
    _extract_features('data_test.csv', 'data_test_feat.csv')


def _extract_features(source: str, target: str):
    """  """
    X = pd.read_csv(os.path.join(const.DATAPATH, source))

    if "isFilter" in X.columns.values:
        X_feat = X[X['isFilter'] is None]
        X_feat = _preprocess(X_feat)
        X_feat = _extract_langfeatures(X_feat)
        X_feat.append(X[X['isFilter'] is not None])
    else:
        X_feat = _preprocess(X)
        X_feat = _extract_basicfeatures(X_feat)
        X_feat = _extract_langfeatures(X_feat)

    X_feat.to_csv(os.path.join(const.DATAPATH, target), index=None)


def _preprocess(X):
    X['words'] = X['sent_words'].progress_apply(lambda x: x.split())
    X['chars'] = X['sent_chars'].progress_apply(lambda x: x.split())
    return X.drop(['sent_chars', 'sent_words'], axis=1)


def _extract_basicfeatures(X):
    """如果没进行规则过滤，则将规则作为特征"""
    # 长度
    X['len'] = X['chars'].progress_apply(lambda x: len(x.split()))
    # 2元模型
    X['pairs'] = X['chars'].progress_apply(lambda x: list(ngrams(x, 2)))

    # 不重复的字数/所有字数 针对 我我我我我和和和和和他过来来来 类型的数据
    X['char_rate'] = X['chars'].progress_apply(lambda x: len(set(x)) / len(x))
    # 不重复的字数/所有字数 针对 我我我我我和和和和和他过来来来 类型的数据
    X['words_rate'] = X['words'].progress_apply(lambda x: len(set(x)) / len(x))
    # 针对二元模型的
    X['pair_rate'] = X['pairs'].progress_apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)

    # 最大的重复字数占比 针对 我X我XX我XXXX我我XX我我XXXX我我我 类型的数据，不一定是连续的
    X['char_mx_rate'] = X['chars'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)
    # 最大的重复字数占比 针对 我我我我我我我我我我 类型的数据
    X['word_mx_rate'] = X['chars'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)
    # 计算是否大量反复，不一定连续地出现某些词组 衡量句子中出现的词语或片段 windows=2, 这是对word的补充
    X['pair_mx_rate'] = X['pairs'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)

    # 计算是否出现大量重复且连续的自
    X['char_rept_mx'] = X['chars'].progress_apply(lambda x: countReptMx(x))
    # 计算是否出现大量重复且连续的词组
    X['word_rept_mx'] = X['words'].progress_apply(lambda x: countReptMx(x))
    X['pair_rept_mx'] = X['pairs'].progress_apply(lambda x: countReptMx(x))

    X.drop(['pairs'], axis=1, inplace=True)

    return X


def _extract_langfeatures(X):
    """  """
    for dname in ['neg', 'pos', 'weibo', 'sms']:
        for dtype, sizes in [('words', [1, 2, 3]), ('chars', [2, 3])]
            for n in sizes:
                path = os.path.join(const.MODELPATH, 'lm_{}_{}_{}.pk'.format(n, dname, dtype))
                if os.path.isfile(path):
                    lm = LanModel(_getPikcle(path), n)
                    X['{}n_ppl_{}_{}'.format(n, dname, dtype)] = X[p].progress_apply(lambda x: lm.perplexity(x) if lm.perplexity(x) != float('inf') else -1)
                    X['{}n_ept_{}_{}'.format(n, dname, dtype)] = X[p].progress_apply(lambda x: lm.entropy(x) if lm.entropy(x) != float('inf') else -1)
                    del lm
                    gc.collect()

    return X


def drop(featlist: List):
    _dropfeatures('data_train_feat.csv', 'data_train_feat.csv', featlist)
    _dropfeatures('data_test_feat.csv', 'data_test_feat.csv', featlist)


def _dropfeatures(source: str, target: str, featlist: List):
    df = pd.read_csv(os.path.join(const.DATAPATH, source))
    X.drop(featlist, axis=1, inplace=True)
    print(X[:5])
    print(X.columns.values.tolist())
    X.to_csv(os.path.join(const.DATAPATH, target), index=None)


def _getPikcle(path: str):
    """  """
    with open(path, 'rb') as f:
        v = pickle.load(f)
    return v
