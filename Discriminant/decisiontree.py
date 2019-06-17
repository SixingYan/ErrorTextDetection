import time
from sklearn.tree import DecisionTreeClassifier as DT


def train(X, y, args):
    print('start...')
    stime = time.time()
    clf = DT(random_state=10)
    clf.fit(X, y)

    return clf


def valid(clf, X, y):
    print('Score={:.4f}'.format(clf.score(X, y)))
