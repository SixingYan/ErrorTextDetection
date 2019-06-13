from SentClassifer import SentenceClassifier

import FilterRules as FR


class ParagraphClassifier(object):
    """docstring for ParagraphClassifier"""

    def __init__(self, sthr:str=None):
        #super(ParagraphClassifier, self).__init__()
        self.split_thr = sthr # 分割的门槛长度
        self.rules = rules
        self.clf = SentenceClassifier()
        self.result = None

    def predict(self, X):
        """
        X 为数据框 数据类型
        """
        if split_thr is not None:
            X = self._reprocess(X)
            
        if self.rules is not None:
            X['target'] = X['sent'].progress_apply(
                lambda x: const.NEG if self._check_rule(x) else None)


        X['target'] = X['sent'].progress_apply()

    def _check_rule(self, x: str):
        for f in self.rules:
            if f(x):
                return True
        return False

    def _reprocess(self, X, y=None):
        pass

    def fit(self, X, y):
        pass
        if self.split_thr is not None:
            X, y = self._reprocess(X, y)

    def score(self, ):
        pass

    def predict(self, ):
        pass

    def predict_component(self, ):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
