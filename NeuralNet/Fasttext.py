import torch
from torch import nn
from torch import Tensor


class Hash


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

        return self._forward()
