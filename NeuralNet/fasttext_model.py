import torch
from torch import nn
from torch import Tensor

from pandas import Series
from import ngrams


class WordCharGramDataset(Dataset):
    """WordCharGramDataset"""

    def __init__(self, words: Series, labels: Seriess):
        # 输入
        self.word_to_idx = {}
        self.wordIdx = words.apply(lambda x: self._word_to_idx(x)).values.tolist()
        self.wordgramIdx = words.apply(lambda x: hashindex(x)).values.tolist()
        self.chargramIdx = words.apply(lambda x: _wordchar_to_idx(x)).values.tolist()
        self.labels = labels.values.tolist()

    def __getitem__(self, i: int):
        return self.wordIdx[i] + self.wordgramIdx[i] + self.chargramIdx[i], self.labels[i]

    def __len__(self):
        return len(self.data)

    def _word_to_idx(self, words: List):
        """  """
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


class FastText(nn.Module):

    def __init__(self, args: object):
        super(FastText, self).__init__()

        self.word_embed = nn.Embedding(args.vocab_size, args.embed_dim)
        self.embed_dim = embed_dim

        # 词字级别的ngram
        self.wordngram_embed = nn.Embedding(args.bucket_size, args.embed_dim)
        self.embed_dim += embed_dim
        self.wordngram_bucket = {}

        hash(x) % args.bucket_size

        if args.ischar is False:
            # 字级别的unigram。如果本身的单位就是字，就不用考虑
            self.charngram_embed = nn.Embedding(args.bucket_size, args.embed_dim)
            self.embed_dim += embed_dim

        self.workflow = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim)
            nn.BatchNorma1d(self.hidden_dim)
            nn.ReLU(inplace=True)
            nn.Linear(hidden_dim, args.label_size)
            nn.Softmax()
        )

    def _forward(self, words, wordngram, charngram=None):
        """
        :param x: LongTensor    Batch_size * Sentence_length

        :return:
        """
        embed = self.(x)

        embed = torch.cat([embed, self.(x)], dim=2)

        if self.:
            embed = torch.cat([embed, self.(x)], dim=2)

        embed = embed.mean(dim=1)

        return self.workflow(embed.view(embed.size(0), -1))  # 输入二维的数据

    def forward(self, x_batch: Tensor):
        """
        :param x: LongTensor    Batch_size * Sentence_length
        :return:
        """
        word
        if self.:

        return self._forward()
