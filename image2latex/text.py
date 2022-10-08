import json
import re
import torch
from torch import Tensor
from abc import ABC, abstractmethod


class Text(ABC):
    def __init__(self):
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

    @abstractmethod
    def tokenize(self, formula: str):
        pass

    def int2text(self, x: Tensor):
        return " ".join([self.id2word[i] for i in x if i > self.eos_id])

    def text2int(self, formula: str):
        return torch.LongTensor([self.word2id[i] for i in self.tokenize(formula)])


class Text100k(Text):
    def __init__(self):
        super().__init__()
        self.id2word = json.load(open("data/vocab/100k_vocab.json", "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.TOKENIZE_PATTERN = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' + "(\w)|" + "(\\\\)"
        )
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens


class Text170k(Text):
    def __init__(self):
        super().__init__()
        self.id2word = json.load(open("data/vocab/170k_vocab.json", "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
        return formula.split()
