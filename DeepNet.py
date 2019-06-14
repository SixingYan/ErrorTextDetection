

df_train = pd.read_csv(os.path.join(const.DATAPATH, ))
df_test = pd.read_csv(os.path.join(const.DATAPATH, ))
df_valid = df_train.sample(frac=0.1)
df_train = df_train.sample(frac=0.9)

# get data type
dtype = 'sent_chars' if args.dtype == 'chars' else 'sent_words'

# get specific model
if args.net == 'fasttext':
    from fasttext_model import *
    tndataset = WordCharGramDataset(df_train[dtype].apply(lambda x: x.split()), df['target'])
    vldataset = WordCharGramDataset(df_valid[dtype].apply(lambda x: x.split()), df['target'])
    tsdataset = WordCharGramDataset(df_test[dtype].apply(lambda x: x.split()), df['target'])
    model = FastText(args)

# get data loader
train_loader, valid_loader test_loader = DataLoader(tndataset, batch_size=5), DataLoader(vldataset, batch_size=5), DataLoader(tsdataset, batch_size=5)

# train
m = train(train_loader, valid_loader, model, args)

# test
evaluate(m, test_loader)

if __name__ == '__main__':
    main()
