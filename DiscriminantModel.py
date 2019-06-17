from sklearn.externals import joblib  # jbolib模块


X, y =

# get specific model
if args.dmodel == 'dt':
    from Discriminant import decisiontree as model

# get data loader
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# train
m = model.train(X_train, y_train, args)


# valid
model.valid(m, X_test, y_test)

# save
joblib.dump(clf, os.path.join(const.MODELPATH, 'tree.model'))

# 读取Model
# clf3 = joblib.load(os.path.join(const.MODELPATH, 'tree.model'))

if __name__ == '__main__':
    main()
