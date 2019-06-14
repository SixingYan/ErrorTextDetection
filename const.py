import os

PATH = os.pwd()

FILENAME = ''

DATAPATH = ''

MODELPATH = ''

POS = 1
NEG = 0


from collections import namedtuple

arguments = {
    'net': "fasttext",
}
ArgsClass = namedtuple('ArgsClass', [k for k in arguments.keys()])
args = ArgsClass(**arguments)
# use as  args.net
