import time
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text
from sklearn.utils import shuffle


def train(X, y):
    X_train, X_test, y_train, y_test = getData(path, source1)
    print('starting...')
    stime = time.time()
    clf = DT(random_state=10)
    clf.fit(X_train, y_train)


def valid(clf, X, y):
    X = pd.read_csv(os.path.join(const.DATAPATH, source))
    X = shuffle(X)

    print('DATA Explore -----------')
    print(X['target'].value_counts())
    y = X['target'].values
    X = X.drop(['target'], axis=1)

    X.describe().to_csv(os.path.join(path, 'describe_{}.csv'.format(source)))

    print('DATA -------------------')
    print(X.columns.values.tolist())

    print('Score={:.4f}'.format(clf.score(X, y)))


def train_valid(source1, source2):
    """ 决策树，就是使用这里的代码 """
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, source1)
    print('starting...')
    stime = time.time()
    clf = DT(random_state=10)
    clf.fit(X_train, y_train)

    tree_text = export_text(clf, feature_names=X_train.columns.values.tolist(), max_depth=20)
    print('Tree Structure : ')
    print(tree_text)

    with open(os.path.join(const.DATAPATH, 'dt_structure_{}.txt'.format(source)), 'w', encoding='utf-8', errors='ignore') as f:
        f.write(tree_text)

    print('Feature importance : ')
    print(clf.feature_importances_)
    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))

    valid(clf, const.DATAPATH, source2)

    return clf
