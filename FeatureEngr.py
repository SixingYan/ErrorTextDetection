import os
import sys
import const

if __name__ == '__main__':
    extract_features()


def extract_features(isBasic=False):
    """  """
    if isBasic:
        X = _extract_basicfeatures(X)
    X = _extract_langfeatures(X)


def _extract_langfeatures(X):

    return X


def _extract_basicfeatures(X):
    return X


def dropfeatures(featlist: List):
    df = pd.read_csv(os.path.join(const.DATAPATH, ))
    return X






    def getData():
        return pd.read_csv(os.path.join(const., ))

    def checkfeats():
    """检查特征数据是否已经存在"""
        if os.path.isfile(os.path.join('')):
            pass
        else:
            raise "no feature data exist!"
        pass

    def checkdata():
    """检查数据是否存在"""
    # raise
        pass

    def fit():
        pass

        checkdata()
        X = getData()

    def corrHeatMap():
        pass

    def featLabelScatter():
        pass

    def main():
    """控制外部输入的参数，来确定运行哪一个方法"""
        pass

    oprt = sys.argv[1]
    if ',' in oprt:
        feat = oprt.split(',')[-1]
        featLabelScatter(feat)
    else:
        if oprt == 'fit':
            fit()
        else:
            corrHeatMap()

    '''
    python FeatureEngr fit
    python FeatureEngr heatmap
    python FeatureEngr scatter,lm_2_neg_jieba
    '''
