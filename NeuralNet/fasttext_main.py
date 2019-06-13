from pandas import Series
from import ngrams


class WordCharGramDataset(Dataset):
    """docstringDataset WordGramHash"""

    def __init__(self, df: Series, labels: Seriess):
        # 已经是列表了
        self.word_to_idx = {}
        self.wordIdx = df.apply(lambda x: self._word_to_idx(x)).values.tolist()
        self.wordgramIdx = df.apply(lambda x: hashindex(x)).values.tolist()
        self.chargramIdx = df.apply(lambda x: _wordchar_to_idx(x)).values.tolist()
        self.labels = labels.values.tolist()

    def __getitem__(self, i: int):
        return self.wordIdx[i] + self.wordgramIdx[i] + self.chargramIdx[i], self.labels[i]

    def __len__(self):
        return len(self.data)

    def _word_to_idx(self, words: List):
        indexs = [None] * len(words)
        for i, w in enumerate(words):
            if w not in self.word_to_idx:
                self.word_to_idx[w] = len(self.word_to_idx)
            indexs[i] = self.word_to_idx[w]
        return indexs

    def _wordchar_to_idx(self, words: List, hashsize: int, gramsize: int):
        chars = []
        for w in words:
            chars += [for tp in ngrams(list(w), gramsize, True)]
        return [hash(tp) % hashsize for tp in chars]

    @staticmethod
    def hashindex(elements: List, hashsize: int, gramsize: int):
        return [hash(tp) % hashsize for tp in ngrams(elements, gramsize, True)]


dataset = WordCharGramDataset(df['sent_jieba'].apply(lambda x: x.split()), df['target'])


train_loader = DataLoader(dataset, batch_size=5)


#


#


#
