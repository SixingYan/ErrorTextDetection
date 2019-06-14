import os
import sys
import const

if __name__ == '__main__':
    extract_features()


def extract_features(isBasic=False):
    """  """
    if "isFilter" in df.columns.values:
        X_feat = X[X['isFilter'] is None]
        X_feat = _preprocess(X_feat)
        X_feat = _extract_langfeatures(X_feat)
        X_feat.append(X[X['isFilter'] is not None])
    else:
        X_feat = _preprocess(X)
        X_feat = _extract_basicfeatures(X_feat)
        X_feat = _extract_langfeatures(X_feat)


def _preprocess(X):
    X['words'] = X['sent_words'].progress_apply(lambda x: x.split())
    X['chars'] = X['sent_chars'].progress_apply(lambda x: x.split())
    return X.drop(['sent_chars', 'sent_words'], axis=1)


def _extract_basicfeatures(X):
    """如果没进行规则过滤，则将规则作为特征"""
    # 长度
    X['len'] = X['chars'].progress_apply(lambda x: len(x.split()))
    # 不重复的字数/所有字数 针对 我我我我我和和和和和他过来来来 类型的数据
    X['char_rate'] = X['chars'].progress_apply(lambda x: len(set(x)))
    # 最大的重复字数占比 针对 我我我我我我我我我我 类型的数据
    X['char_mx_rate'] = X['chars'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)

    # 衡量句子中出现的词语或片段 windows=2

    这地方有些不对！

    X['pairs'] = X['chars'].progress_apply(lambda x: list(ngrams(x, 2)))
    X['pair_rate'] = X['pairs'].progress_apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
    X['pair_mx_rate'] = X['pairs'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)

    X['rept_mx'] = X['words'].progress_apply(lambda x: countReptMx(x))
    X['rept_mx_rate'] = X['rept_mx'] / X['len']
    X['rept_mx_2'] = X['pairs'].progress_apply(lambda x: countReptMx(x))
    X.drop(['pairs'], axis=1, inplace=True)

    return X


def _extract_langfeatures(X):
    """  """
    for dname in ['neg', 'pos', 'weibo', 'sms']:
        for dtype, sizes in [('words', [1, 2, 3]), ('chars', [2, 3])]
            for n in sizes:
                if os.path.isfile():
                    lm = LanModel(getPickle(), n)
                    X['{}n_ppl_{}_{}'.format(n, dname, dtype)] = X[p].progress_apply(lambda x: lm.perplexity(x) if lm.perplexity(x) != float('inf') else -1)
                    X['{}n_ept_{}_{}'.format(n, dname, dtype)] = X[p].progress_apply(lambda x: lm.entropy(x) if lm.entropy(x) != float('inf') else -1)
                    del lm
                    gc.collect()

    return X


def dropfeatures(featlist: List):
    df = pd.read_csv(os.path.join(const.DATAPATH, data))
    
    return X

    '''
    python FeatureEngr fit
    python FeatureEngr heatmap
    python FeatureEngr scatter,lm_2_neg_jieba
    '''
