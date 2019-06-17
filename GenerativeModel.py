

import Generative


getPickle()

if os.path.isfile(os.path.join(const.MODELPATH, 'data_train_{}_{}.vec'.format(args.dtype, args.vectype))) and os.path.isfile(os.path.join(const.MODELPATH, 'data_test_{}_{}.vec'.format(args.dtype, args.vectype))):
    train = getPickle(os.path.join(const.MODELPATH, 'data_train_{}_{}.vec'.format(args.dtype, args.vectype)))
    X_test = getPickle(os.path.join(const.MODELPATH, 'data_train_{}_{}.vec'.format(args.dtype, args.vectype)))

    df_train = pd.read_csv(os.path.join(const.DATAPATH, 'data_train.csv'))
    X_train, X_valid, y_train, y_valid = (train, df_train['target'].values.tolist(), test_size=0.1, shuffle=True)
    y_test = pd.read_csv(os.path.join(const.DATAPATH, 'data_test.csv'))['target'].values.tolist()

# get specific model
if args.generative == 'svm':
    from Discriminant import svm as model

# train
m = model.train(X_train, y_train, X_valid, y_valid, args)

# valid
model.valid(m, X_test, y_test)

# save
joblib.dump(clf, os.path.join(const.MODELPATH, 'tree.model'))


if __name__ == '__main__':
    main()
