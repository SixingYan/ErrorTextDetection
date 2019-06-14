# encoding = utf-8
import re
from typing import List
import pandas as pd
import os
import jieba

import const


def preprocess(source: str='data.csv'):
    """  """
    df = pd.read_csv(os.path.join(const.DATAPATH, source))
    df['sent'] = df['sent'].apply(lambda x: str(x))
    df['sent'] = df['sent'].apply(lambda x: x.encode('utf-8').decode('utf-8'))

    df = df[['sent', 'target']]
    df['sent_chars'] = df['sent'].apply(lambda x: ' '.join(_parse_single(x)))
    df['sent_words'] = df['sent'].apply(lambda x: ' '.join(w for w in jieba.cut(x)))
    print(df.head())
    # train and test set
    target = 'data_{}.csv'
    df.sample(frac=0.2).to_csv(os.path.join(const.DATAPATH, target.format('test')), index=None)
    df.sample(frac=0.8).to_csv(os.path.join(const.DATAPATH, target.format('train')), index=None)


def _parse_single(sent: str)->List:
    """ 把中文句子转化成字的列表，这里只保留中文字符 """
    sent = ''.join(w.strip() for w in re.findall(r'[\u4e00-\u9fa5]', sent) if len(w.strip()) > 0)
    return [w.strip() for w in sent if len(w.strip()) > 0]


def main():
    preprocess()


if __name__ == '__main__':
    main()
