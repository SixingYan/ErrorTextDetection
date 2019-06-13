'''
命名规则：
1. 规则函数前面加 _
2. is 开头表示 计数型 的函数
3. has 开头表示 占比型 的函数

TO-DO:
1. 允许输入自定义阈值
2. 分别统计每个过滤规则的正确率
3. 设置统一入口，从外部控制运行哪一个函数
'''
from typing import List
import pandas as pd

if __name__ == '__main__':
    filtering('', '')
    # exploring()
    # noFiltering()

def noFiltering():
    """"""
    df = pd.read_csv(os.path.join(const.DATAPATH, source))

    df['predict'] = None

    df.to_csv(os.path.join(const.DATAPATH, target), index=None)
    

def exploring(source, thrList: List=None):
    def countCor(sent, chars, words, target):
        if toFilter(sent, chars, words) == (True if target == const.NEG else False):
            return 1
        else:
            return 0

    df = pd.read_csv(os.path.join(const.DATAPATH, source))

    df['filter_result'] = df.progress_apply(lambda x: countCor(x.sent, x.chars, x.words, x.target))

    accurancy = sum(df['filter_result']) / df.shape[0]

    print('Filtering accurancy is {:.4f}'.format(accurancy))


def filtering(source, target):
    pass

    def check(sent, chars, words, target)->bool:
        if toFilter(sent, chars, words) is True:
            return const.NEG
        return None

    df = pd.read_csv(os.path.join(const.DATAPATH, source))

    df['predict'] = df.progress_apply(lambda x: check(x.sent, x.chars, x.words, x.target))

    df.to_csv(os.path.join(const.DATAPATH, target), index=None)


def toFilter(sent, chars, words, thrList: List=None)->bool:
    """唯一对外的方法"""
    res = False

    if _isLen(sent) is True:
        return res
    elif _hasCnRate(sent) is True:
        return res
    elif _isReaptChars(chars) is True:
        return res
    elif _hasReatChars(chars) is True:
        return res
    elif _isReaptWord(words) is True:
        return res
    elif _hasReatWord(words) is True:
        return res
    elif _isDuplWord(words) is True:
        return res
    elif _hasDuplWord(words) is True:
        return res
    elif _isNoCN(sent) is True:
        return res

    return res


def _isLen(sent: str, thr: int=3)->bool:
    """"""
    return True if len(sent) < thr else False


def _hasCnRate(sent: str, thr: int=0.3):
    """"""
    return True if sum(1 for _ in re.findall(r'[\u4e00-\u9fa5]', sent)) / len(sent) < thr else False


def _isReaptChars(chars: List, thr: int=7)->bool:
    """ 计算出现连续单字最大长度 """
    return True if _countReptMx(words) > thr else False


def _hasReatChars(chars: List, thr: int=0.5)->bool:
    """"""
    return False if len(chars) == 0 or _countReptMx(chars) / len(chars) <= thr else True


def _isReaptWord(words: List, thr: int=5)->bool:
    """连续重复的词语"""
    return True if _countReptMx(words) > thr else False


def _hasReatWord(words: List, thr: int=0.5)->bool:
    return False if len(words) == 0 or _countReptMx(words) / len(words) <= thr else True


def _isDuplWord(words: List, thr: int=5):
    """大量重复的词语"""
    return True if Counter(words).most_common(1)[0][1] > thr else False


def _hasDuplWord(words: List, thr: int=0.3):
    return False if len(words) == 0 or Counter(words).most_common(1)[0][1] / len(words) <= thr else True


def _isNoCN(sent: str, thr: int=10)->bool:
    """含有过多非中文字符"""
    return True if len(sent) - sum(1 for _ in re.findall(r'[\u4e00-\u9fa5]', sent)) > thr else False


def _countReptMx(strList: List)->int:
    """  """
    mx, c, pre = 1, 1, None
    for i, w in enumerate(strList):
        if pre is None:
            pre = w
            continue
        if w == pre:
            c += 1
        else:
            if c > mx:
                mx = c
            pre = w
            c = 1
        if i == len(strList) - 1:
            if c > mx:
                mx = c
    return mx
