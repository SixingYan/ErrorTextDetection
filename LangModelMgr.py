import pickle
import time
import pandas as pd
import os
from typing import List

from nltk.util import ngrams
from nltk.lm import MLE
# from nltk.lm.preprocessing import padded_everygram_pipeline

import const


class LanModel(object):
    """ 封装了ngrams和计算方法 """

    def __init__(self, lm, n):
        self.lm = lm
        self.n = n

    def entropy(self, chars: List)->float:
        return self.lm.entropy(ngrams(chars, self.n,
                                      True, True, '<s>', '</s>'))

    def perplexity(self, chars: List)->float:
        return self.lm.perplexity(ngrams(chars, self.n,
                                         True, True, '<s>', '</s>'))


def _train(n: int, texts: List):
    """ texts 已经分词的文本列表"""
    lm = MLE(n)

    train, vocab = [], set([])
    for t in texts:
        g = ngrams(t, n, pad_left=True, pad_right=True,
                   left_pad_symbol='<s>', right_pad_symbol='</s>')
        g = list(g)
        vocab = vocab | set(t)
        train.append(g)

    lm.fit(train, vocabulary_text=list(vocab))
    return lm


def _save(m, col: str, n: int, corp: str):
    with open(os.path.join(const.MODELPATH, 'lm_{}_{}_{}.pk'.format(n, corp, col)), 'wb') as f:
        pickle.dump(m, f)


def batch_train(source: str='data_train.csv'):
    """ """
    df = pd.read_csv(os.path.join(path, source), encoding='UTF-8')

    if "isFilter" in df.columns.values:  # 判断是否有过滤
        df = df[df["isFilter"] is None]

    df_neg = df[df['target'] == const.NEG]
    df_pos = df[df['target'] == const.POS]
    for i, dataf in enumerate([df_neg, df_pos]):
        for col, sizes in [('sent_words', [1, 2, 3]), ('sent_chars', [2, 3])]:
            dataf[col] = dataf[col].apply(lambda x: str(x))
            dataf[col] = dataf[col].apply(lambda x: x.split())
            for s in sizes:
                print('start n={} dtype={} target={}'.format)
                stime = time.time()
                lm = _train(s, df[col].values.tolist())
                _save(lm, col, s, 'neg' if i == const.NEG else 'pos')
                print('Time cost={:.4f} Vocab size={}'.format((time.time() - stime) / 60, len(lm.vocab)))


def main():
    batch_train()


if __name__ == '__main__':
    main()
